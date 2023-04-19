#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h> 
#include <sys/socket.h>
#include <arpa/inet.h>
#include "hiredis.h"
#include "brand.h"


// Cerebus packet definition, adapted from the standard Blackrock library
// https://github.com/neurosuite/libcbsdk/cbhwlib/cbhwlib.h
typedef struct cerebus_packet_header_t {
    uint32_t time;
    uint16_t chid;
    uint8_t type;
    uint8_t dlen;
} cerebus_packet_header_t;


// this allows for only a max streams based on constant above
// change that constant if you want
typedef struct graph_parameters_t {
    char input_stream_name[30];
    char output_stream_name[30];
    char coefs_stream_name[30];
    int samp_freq;
    int chan_per_stream;
    int samp_per_stream;
    int chan_total;
    int start_channel;
} graph_parameters_t;

// intialize support functions
void initialize_signals();
void initialize_parameters(redisContext *c, graph_parameters_t *p);
void handler_SIGINT(int exitStatus);
void shutdown_process();
void initialize_coefficients(redisContext *c, double **coefs, graph_parameters_t *p);

char NICKNAME[] = "reReference";
char SUPERGRAPH_ID[256];

int flag_SIGINT = 0;

redisReply *reply;
redisContext *redis_context;


int main (int argc_main, char **argv_main) {

    initialize_signals();

    redis_context = parse_command_line_args_init_redis(argc_main, argv_main, NICKNAME);  
    emit_status(redis_context, NICKNAME, NODE_STARTED, NULL);

    graph_parameters_t graph_parameters = {};
    initialize_parameters(redis_context, &graph_parameters); 

    int samp_freq       = graph_parameters.samp_per_stream;
    int samp_per_stream = graph_parameters.samp_per_stream;
    int chan_per_stream = graph_parameters.chan_per_stream;

    // initialize re-referencing matrix
    double **coefs = (double**)malloc(chan_per_stream * sizeof(double*));
    for (int i = 0; i < chan_per_stream; i++) {
        coefs[i] = (double*)malloc(chan_per_stream * sizeof(double));
    }
    initialize_coefficients(redis_context, coefs, &graph_parameters);

    // argc    : The number of arguments in argv. The calculation is:
    //           int argc = 3 + 2 * 4;
    //           3                  -> xadd cerebusAdapter *
    //           4 x 2              -> timestamps (3 types) and sample array x (key, value) pairs
    // argvlen : The length of the strings in each argument of argv
    //           This will be an array of pointers since we've got a 
    //           variable number of streams
    //
    // We will be using the same argc and indices for all different frequencies,
    // but we will use different argv and argvlen for each.
    
    int argc        = 3 + (2 * 4); // argcount = xadd + key:value for everything else
    size_t *argvlen = malloc(argc * sizeof(size_t));

    // argv : This contains the arguments to be executed by redis. 
    //        the argv has the form:
    //        xadd cerebusAdapter * num_samples [string] timestamps [int32] samples [int16] ... 
    //        We begin by populating the entries manually
    //        Starting at index position [3], we start adding the key data, always of form key [value]
    //        So that the key identifier (i.e. the string) is an odd number and the value is even
    //        This format is the same for every frequency of Redis data, so we don't need to 
    //        change the index locations etc
    
    // We keep track of the indexes. Each ind_ variable keeps track of where the (key value) begins
     
    int ind_xadd                        = 0;                         	// xadd cerebusAdapter *
    int ind_cerebus_timestamps          = ind_xadd + 3;              	// timestamps [data]
    int ind_current_time                = ind_cerebus_timestamps + 2;   // current_time [data]
    int ind_udp_received_time           = ind_current_time + 2;      	// udp_received_time [data]
    int ind_samples                     = ind_udp_received_time + 2;    // samples [data array] 
    
    // Now allocatie memory

    int len = 16;
    char *argv[argc];

    // xadd cerebusAdapter *
	for (int i = 0; i < ind_cerebus_timestamps; i++) {
		argv[i] = malloc(len);
	} 
    // timestamps keys and values
    argv[ind_cerebus_timestamps]     = malloc(len);
    argv[ind_cerebus_timestamps + 1] = malloc(sizeof(int32_t) * samp_per_stream);
    argv[ind_current_time]           = malloc(len);
    argv[ind_current_time + 1]       = malloc(sizeof(struct timespec) * samp_per_stream);
    argv[ind_udp_received_time]      = malloc(len);
    argv[ind_udp_received_time + 1]  = malloc(sizeof(struct timeval) * samp_per_stream);
    argv[ind_samples]                = malloc(len);
    argv[ind_samples + 1]            = malloc(sizeof(double) * samp_per_stream * chan_per_stream);

    // populating the argv strings
	argvlen[0] = sprintf(argv[0], "%s", "xadd"); // write the string "xadd" to the first position in argv, and put the length into argv
	argvlen[1] = sprintf(argv[1], "%s", graph_parameters.output_stream_name); //stream name
	argvlen[2] = sprintf(argv[2], "%s", "*");
    // and the key/values
    argvlen[ind_cerebus_timestamps]  = sprintf(argv[ind_cerebus_timestamps]  , "%s", "timestamps");
    argvlen[ind_current_time]        = sprintf(argv[ind_current_time]  , "%s", "BRANDS_time");
    argvlen[ind_udp_received_time]   = sprintf(argv[ind_udp_received_time]  , "%s", "udp_recv_time");
    argvlen[ind_samples]             = sprintf(argv[ind_samples], "%s", "samples");

    printf("[%s] Starting main loop...\n", NICKNAME);

    struct timespec current_time;
    struct timeval udp_received_time;

    char last_redis_id [30];
    strcpy(last_redis_id, "$");
    char redis_string[256] = {0};
    char *redis_data_nsp_timestamps; 
    char *redis_data_udp_timestamps; 
    char *redis_data_samples;        

    int16_t sample_temp;
    double samples[chan_per_stream][samp_per_stream];
    double samples_reref[chan_per_stream][samp_per_stream];
    //double samples_ref_temp;

    while (1) {

        //if (flag_SIGINT) 
        //    shutdown_process();
        
        freeReplyObject(reply); 
        // Read new samples from redis stream
        sprintf(redis_string, "xread count 1 block 0 streams %s %s", graph_parameters.input_stream_name, last_redis_id);
        reply = redisCommand(redis_context, redis_string);

        // The xread value is nested:
        // dim0 [0] The first stream (input_stream)
        // dim1 [1] Stream data
        // dim2 [0+] The stream samples we're getting data from
        // dim3 [0] The redis timestamp
        // dim3 [1] The data content from the stream
        // dim4 [1] Cerebus timestamp from the stream
        // dim4 [3] Monotonic timestamp from the stream
        // dim4 [5] UDP received time from the stream
        // dim4 [7] The continuous data content from the stream

        // Save timestamp/id of last redis sample
        strcpy(last_redis_id, reply->element[0]->element[1]->element[0]->element[0]->str);  

        redis_data_nsp_timestamps = reply->element[0]->element[1]->element[0]->element[1]->element[1]->str;
        redis_data_udp_timestamps = reply->element[0]->element[1]->element[0]->element[1]->element[5]->str;
        redis_data_samples        = reply->element[0]->element[1]->element[0]->element[1]->element[7]->str;

        for(int n = 0; n < samp_per_stream; n++)
        {         
            memcpy(&argv[ind_cerebus_timestamps + 1][n * sizeof(uint32_t)],      
                &redis_data_nsp_timestamps[n * sizeof(uint32_t)],  
                sizeof(uint32_t));
            memcpy(&argv[ind_udp_received_time  + 1][n * sizeof(struct timeval)],   
                &redis_data_udp_timestamps[n * sizeof(struct timeval)],   
                sizeof(struct timeval)); 
            // could be done without the loop, but would require transposing the data    
            for (int iChan = 0; iChan < chan_per_stream; iChan++)
            {
                memcpy(&sample_temp,      
                    &redis_data_samples[(n + iChan*samp_per_stream) * sizeof(int16_t)],
                    sizeof(int16_t));
                samples[iChan][n] = (double)sample_temp;
            }  
        }

        memset(samples_reref, 0, chan_per_stream * samp_per_stream * sizeof(double));
        // A linear algebra package for this matrix operation should be waaay more efficient than this
        //#pragma unroll 
        for(int n = 0; n < samp_per_stream; n++)
        {         
            //#pragma unroll 
            for (int iChan = 0; iChan < chan_per_stream; iChan++)
            {
                // Compute reference from weighted sum of all channels
                //samples_ref_temp = 0;
                //samples_reref[iChan][n] = 0;

                //#pragma unroll 
                for (int jChan = 0; jChan < chan_per_stream; jChan++)
                {
                    //samples_ref_temp += samples[jChan][n] * coefs[iChan][jChan];
                    samples_reref[iChan][n] += samples[jChan][n] * coefs[iChan][jChan];
                }
                // Subtract reference from each channel
                //samples_reref[iChan][n] = samples[iChan][n] - samples_ref_temp;
            }
        }

        // This gets the current system time
        clock_gettime(CLOCK_MONOTONIC, &current_time);

        for(int n = 0; n < samp_per_stream; n++)
        {         
            memcpy(&argv[ind_current_time       + 1][n * sizeof(struct timespec)],      
                &current_time,  
                sizeof(struct timespec));
            for (int iChan = 0; iChan < chan_per_stream; iChan++)
            {
                //sample_temp = samples_reref[iChan][n];
                memcpy(&argv[ind_samples + 1][(n + iChan*samp_per_stream) * sizeof(double)],      
                    &samples_reref[iChan][n],
                    sizeof(double));
            }  
        }

        // Update argvlen
        argvlen[ind_cerebus_timestamps + 1]      = sizeof(int32_t) * samp_per_stream;
        argvlen[ind_current_time + 1]            = sizeof(struct timespec) * samp_per_stream;
        argvlen[ind_udp_received_time + 1]       = sizeof(struct timeval) * samp_per_stream;
        argvlen[ind_samples + 1]                 = sizeof(double) * samp_per_stream * chan_per_stream;

        // Write to Redis
        freeReplyObject(redisCommandArgv(redis_context,  argc, (const char**) argv, argvlen));
    }

    // We should never get here, but we should clean our data anyway.
    for (int ii = 0; ii < argc; ii++) {
        free(argv[ii]);
    }
    free(argvlen);
    return 0;
}

//------------------------------------
// Initialization functions
//------------------------------------

void initialize_coefficients(redisContext *c, double **coefs, graph_parameters_t *p)
{
        int chan_per_stream = p->chan_per_stream;
        int chan_total = p->chan_total;
        int start_channel = p->start_channel;
        
        freeReplyObject(reply); 
        // Read new samples from redis stream
        //sprintf(redis_string, );
        reply = redisCommand(c, "xrevrange %s + - COUNT 1", p->coefs_stream_name);
        
        if (reply == NULL || reply->type != REDIS_REPLY_ARRAY || reply->elements == 0)
        {          
            printf("[%s] Coefficients entry not found in Redis. Setting coefficients to compute mean of all channels.\n", NICKNAME);

            for (int iChan = 0; iChan < chan_per_stream; iChan++)
            {
                for (int jChan = 0; jChan < chan_per_stream; jChan++)
                {
                    //coefs[iChan][jChan] = 1.0/chan_per_stream;
                    if (iChan == jChan)
                        coefs[iChan][jChan] = 1.0 - 1.0/chan_per_stream;
                    else
                        coefs[iChan][jChan] = -1.0/chan_per_stream; 
                }  
            }  
        }
        else
        {
            printf("[%s] Coefficients entry found in Redis stream: %s.\n", NICKNAME, p->coefs_stream_name);

            char *redis_coef_samples; 
            double coef_temp;

            // The xrevrange value is nested:
            // dim0 [0+] The stream samples we're getting data from
            // dim1 [0] The redis timestamp
            // dim1 [1] The data content from the stream
            // dim2 [1] Coefficient values
            redis_coef_samples = reply->element[0]->element[1]->element[1]->str;

            for(int iChan = 0; iChan < chan_per_stream; iChan++)
            {         
                for (int jChan = 0; jChan < chan_per_stream; jChan++)
                {
                    memcpy(&coef_temp,      
                        &redis_coef_samples[((jChan+start_channel) + (iChan+start_channel)*chan_total) * sizeof(double)],
                        sizeof(double));
                    //coefs[iChan][jChan] = coef_temp;
                    if (iChan == jChan)
                        coefs[iChan][jChan] = 1.0 - coef_temp;
                    else
                        coefs[iChan][jChan] = -coef_temp; 
                }  
            }
        }

    printf("[%s] Coefficients loaded.\n", NICKNAME);
}

void initialize_signals() {

    printf("[%s] Attempting to initialize signal handlers.\n", NICKNAME);

    signal(SIGINT, &handler_SIGINT);

    printf("[%s] Signal handlers installed.\n", NICKNAME);
}

void initialize_parameters(redisContext *c, graph_parameters_t *p)
{
    printf("[%s] Loading graph parameters\n", NICKNAME);
    
    // Initialize Supergraph_ID 
    SUPERGRAPH_ID[0] = '0';
    // Now fetch data from the supergraph and populate entries
    redisReply *reply_supergraph = NULL;
    const nx_json *supergraph_json = get_supergraph_json(c, reply_supergraph, SUPERGRAPH_ID); 
    if (supergraph_json == NULL) {
        emit_status(c, NICKNAME, NODE_FATAL_ERROR,
                    "No supergraph found for initialization. Aborting.");
        exit(1);
    }
    strcpy(p->input_stream_name, get_parameter_string(supergraph_json, NICKNAME , "input_stream_name"));
    strcpy(p->output_stream_name, get_parameter_string(supergraph_json, NICKNAME , "output_stream_name"));
    strcpy(p->coefs_stream_name, get_parameter_string(supergraph_json, NICKNAME , "coefs_stream_name"));
    p->samp_freq = get_parameter_int(supergraph_json, NICKNAME , "samp_freq");
    p->chan_per_stream = get_parameter_int(supergraph_json, NICKNAME , "chan_per_stream");
    p->samp_per_stream = get_parameter_int(supergraph_json, NICKNAME , "samp_per_stream");
    p->chan_total = get_parameter_int(supergraph_json, NICKNAME , "chan_total");
    p->start_channel = get_parameter_int(supergraph_json, NICKNAME , "start_channel");

    printf("[%s] Initialization complete. Reading from stream: %s. Writing to stream: %s\n", NICKNAME, p->input_stream_name, p->output_stream_name);

    // Free memory, since all relevant information has been transffered to the parameter struct at this point
    // using memcpy and strcpy commands
    nx_json_free(supergraph_json);
    freeReplyObject(reply_supergraph);
}


void shutdown_process() {
    printf("[%s] SIGINT received. Shutting down.\n", NICKNAME);

	printf("[%s] Shutting down redis.\n", NICKNAME);

    freeReplyObject(reply); 
	redisFree(redis_context);

	printf("[%s] Exiting.\n", NICKNAME);

    exit(0);
}

//------------------------------------
// Handler functions
//------------------------------------

void handler_SIGINT(int exitStatus) {
    flag_SIGINT++;

    shutdown_process();
}