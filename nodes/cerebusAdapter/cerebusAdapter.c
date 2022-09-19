/* cerebusAdapter.c
 * Converts cerebrus generic packets to the UDP stream
 *
 * The goal of this process is to convert UDP information to a Redis stream. The
 * stream has the following format:
 *
 * cerebusAdapter num_samples [string] timestamps [uint32 binary array] samples [int16 binary array] 
 * 
 * It makes use of the redisCommandArgv function, which has the form: (redis_context,  argc, (const char**) argv, argvlen));
 *
 * Here, argc is the number of strings being sent. neural_argv is the content of the string, and argvlen is the string lengths
 * for the strings. Note that Redis is binary safe, so we can store raw binaries nicely if we want.
 *
 * This function sits and blocks on a udp socket. When a new packet arrives it then creates a pointer
 * to the point of the UDP payload that we would expect to be a cerebus packet header. 
 * If it has the right data type, it copies the data from the UDP payload to populate neural_argv.
 * It keeps track of the neural_argvlen prior to submission to Redis.
 *
 * When sufficient samples have been collected (defined in the cerebusAdapter.yaml file) it then
 * writes the collected neural_argv to Redis and then starts again.
 *
 * One thing to keep in mind is that the keys for the stream will always be even numbers, and the
 * corresponding values will be odd. That's why there's so much +1 notation everywhere.
 *
 * David Brandman
 * June 2020
 * Version 0.1
 */



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h> 
#include <sys/socket.h>
#include <arpa/inet.h>
#include "tools_CParser.h"
#include "hiredis.h"

#define MAXSTREAMS 10 // maximum number of streams allowed -- to set up all of the arrays as necessary


// Cerebrus packet definition, adapted from the standard Blackrock library
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
    int broadcast_port;
    int num_streams;
    char *stream_names[MAXSTREAMS];
    int samp_freq[MAXSTREAMS];
    int packet_type[MAXSTREAMS];
    int chan_per_stream[MAXSTREAMS];
    int samp_per_stream[MAXSTREAMS];
} graph_parameters_t;


// intialize support functions
void initialize_redis(char *yaml_path);
void initialize_signals();
int  initialize_socket(char *yaml_path);
//void initialize_parameters(yaml_parameters_t *p);
void initialize_parameters(graph_parameters_t *p, char *yaml_path);
void parameter_array_parser(int is_char, char *in_string, void *array_ind);
void initialize_realtime(char *yaml_path);
void handler_SIGINT(int exitStatus);
void shutdown_process();
void print_argv(int, char **, size_t *);

char PROCESS[] = "cerebusAdapter";

redisContext *redis_context;

int flag_SIGINT = 0;



int main (int argc_main, char **argv_main) {

    // initializing the input arguments -- should give us the yaml name
    if(argc_main<2){
        printf("Please supply a path to a yaml file");
        return 1;
    }

    char *yaml_path = argv_main[1];


    //debugging file output

    initialize_redis(yaml_path);

    initialize_signals();

    // Uncommenting this results in bash fork error since cerebusAdapter uses Redis
    //initialize_realtime();

    int udp_fd = initialize_socket(yaml_path);

    //yaml_parameters_t yaml_parameters = {0};
    graph_parameters_t graph_parameters;
    initialize_parameters(&graph_parameters, yaml_path); // this is a little more complicated with a var num IOs
    int numStreams = graph_parameters.num_streams; //will use this a lot, so pull it out


    // argc    : The number of arguments in argv. The calculation is:
    //           int argc = 3 + 2 * 4;
    //           3                  -> xadd cerebusAdapter *
    //           4                  -> timestamps (3 types) and sample array
    //           2                  -> (key, value) pairs
    // argvlen : The length of the strings in each argument of argv
    //           This will be an array of pointers since we've got a 
    //           variable number of streams
    //
    // We will be using the same argc and indices for all different frequencies,
    // but we will use different argv and argvlen for each.
    
    int argc        = 3 + (2 * 4); // argcount = xadd + key:value for everything else

    size_t *argvlen[numStreams];
    for ( int ii = 0; ii < numStreams; ii++) {
        argvlen[ii] = malloc(argc * sizeof(size_t)); // arvlen (length of each argv entry)
    }


    // argv : This contains the arguments to be executed by redis. 
    //        the argv has the form:
    //        xadd cerebusAdapter * num_samples [string] timestamps [int32] samples [int16] ... 
    //        We begin by populating the entries manually
    //        Starting at index position [3], we start adding the key data, always of form key [value]
    //        So that the key identifier (i.e. the string) is an odd number and the value is even
    //        This format is the same for every frequency of Redis data, so we don't need to 
    //        change the index locations etc
    
    // We keep track of the indexes. Each ind_ variable keeps track of where the (key value) begins
    //
     

    int ind_xadd                        = 0;                         	// xadd cerebusAdapter *
    int ind_cerebus_timestamps          = ind_xadd + 3;              	// timestamps [data]
    int ind_current_time                = ind_cerebus_timestamps + 2;   // current_time [data]
    int ind_udp_received_time           = ind_current_time + 2;      	// udp_received_time [data]
    int ind_samples                     = ind_udp_received_time + 2;    // samples [data array] 
    
    //////////////////////////////////////////
    // Now we begin the arduous task of allocating memory. We want to be able to hold
    // data of types strings, int16 and int32, so we need to be careful.

    int len = 16;
    char **argvPtr[numStreams];
    for (int ii = 0; ii < numStreams; ii++) {
        //char *argv[argc];
        argvPtr[ii] = malloc(argc * sizeof(size_t));
        int samp_per_stream = graph_parameters.samp_per_stream[ii];
        int chan_per_stream = graph_parameters.chan_per_stream[ii];


        // space for xadd streamName *
        for (int jj = 0; jj < ind_cerebus_timestamps; jj++) {
            argvPtr[ii][jj] = malloc(len);
        }
        
        // allocating memory for timestamps [data]
        argvPtr[ii][ind_cerebus_timestamps]             = malloc(len);
        argvPtr[ii][ind_cerebus_timestamps + 1]         = malloc(sizeof(int32_t) * samp_per_stream);
        
        // allocating memory for current_time [data]
        argvPtr[ii][ind_current_time]                   = malloc(len);
        argvPtr[ii][ind_current_time + 1]               = malloc(sizeof(struct timeval) * samp_per_stream);

        // allocating memory for udp_received_time [data]
        argvPtr[ii][ind_udp_received_time]              = malloc(len);
        argvPtr[ii][ind_udp_received_time + 1]          = malloc(sizeof(struct timeval) * samp_per_stream);
   
        // allocating memory for samples:  [data0 ... dataX]
        argvPtr[ii][ind_samples]                        = malloc(len);
        argvPtr[ii][ind_samples + 1]                    = malloc(sizeof(int16_t) * samp_per_stream * chan_per_stream);
 
        // At this point we start populating neural_argv strings
        // Start by adding xadd cerebusAdapter *
        // And then add the keys for num_samples, timestamps, channel list, and sample array

        argvlen[ii][0] = sprintf(argvPtr[ii][0], "%s", "xadd");
        argvlen[ii][1] = sprintf(argvPtr[ii][1], "%s", graph_parameters.stream_names[ii]);
        argvlen[ii][2] = sprintf(argvPtr[ii][2], "%s", "*");
        
        argvlen[ii][ind_cerebus_timestamps]      = sprintf(argvPtr[ii][ind_cerebus_timestamps]  , "%s", "timestamps");
        argvlen[ii][ind_current_time]            = sprintf(argvPtr[ii][ind_current_time]  , "%s", "BRANDS_time");
        argvlen[ii][ind_udp_received_time]       = sprintf(argvPtr[ii][ind_udp_received_time]  , "%s", "udp_recv_time");
        argvlen[ii][ind_samples]                 = sprintf(argvPtr[ii][ind_samples], "%s", "samples");
        

        //argvPtr[ii] = malloc(sizeof(argv));
        //memcpy(argvPtr[ii], argv, sizeof(argv));
        //free(argv);
    }



    printf("[%s] Entering loop...\n", PROCESS);
    
    // How many samples have we copied 
    int n[numStreams];  
    for(int ii = 0; ii < numStreams; ii++)
        n[ii] = 0;

    // We use rcvmsg because we want to know when the kernel received the UDP packet
    // and because we want the socket read to timeout, allowing us to gracefully
    // shutdown with a SIGINT call. Using recvmsg means there's a lot more overhead
    // in actually getting to business, as seen below

    char *buffer = malloc(65535); // max size of conceivable packet
    char msg_control_buffer[2000] = {0};
    
    struct iovec message_iovec = {0};
    message_iovec.iov_base = buffer;
    message_iovec.iov_len  = 65535;

    struct msghdr message_header = {0};
    message_header.msg_name       = NULL;
    message_header.msg_iov        = &message_iovec;
    message_header.msg_iovlen     = 1;
    message_header.msg_control    = msg_control_buffer;
    message_header.msg_controllen = 2000;

    struct timeval current_time;
    struct timeval udp_received_time;
    struct cmsghdr *cmsg_header; // Used for getting the time UDP packet was received

    while (1) {

        int udp_packet_size = recvmsg(udp_fd, &message_header, 0);

        if (flag_SIGINT) 
            shutdown_process();

        // The timer has timed out or there was an error with the recvmsg() call
        if (udp_packet_size  <= 0) {
            printf("[%s] timer has timed out or there was an error with the recvmsg() call!\n",PROCESS);
            continue;
        }

        // For convenience, makes it much easier to reason about
        char *udp_packet_payload = (char*) message_header.msg_iov->iov_base;
        
        // These two lines of code get the time the UDP packet was received
        cmsg_header = CMSG_FIRSTHDR(&message_header); 
        memcpy(&udp_received_time, CMSG_DATA(cmsg_header), sizeof(struct timeval));



        // We know that the UDP packet is organized as a series
        // of cerebus packets, so we're going to read them in sequence and decide what to do
        // cb_packet_ind is the index of the start of the cerebus packet we're reading from.
        
        int cb_packet_ind = 0; 
        while (cb_packet_ind <= udp_packet_size) {

            // First check: can we safely read the remaining payload content. We should
            // have at least sizeof(cerebus_packet_header_t) bytes left to read
            // If not something went wrong and we go fetch the next packet

            if (cb_packet_ind + sizeof(cerebus_packet_header_t) > udp_packet_size) {
                break;
            }


            // Create a pointer to the cerebus packet at the current location of the udp payload
            cerebus_packet_header_t *cerebus_packet_header = (cerebus_packet_header_t*) &udp_packet_payload[cb_packet_ind];

            // for each stream, check if there's the relevant packet type being pulled in 
            for (int iStream = 0; iStream < numStreams; iStream++){
                if (cerebus_packet_header->type == graph_parameters.packet_type[iStream]) {
                    
                    // This gets the current system time
                    gettimeofday(&current_time,NULL);
    
                    // Copy the timestamp information into argvPtr
                    memcpy( &argvPtr[iStream][ind_cerebus_timestamps + 1][n[iStream] * sizeof(uint32_t)],      &cerebus_packet_header->time,  sizeof(uint32_t));
                    memcpy( &argvPtr[iStream][ind_current_time + 1][n[iStream] * sizeof(struct timeval)],      &current_time,                 sizeof(struct timeval));
                    memcpy( &argvPtr[iStream][ind_udp_received_time + 1][n[iStream] * sizeof(struct timeval)], &udp_received_time,            sizeof(struct timeval));
    
                    // The index where the data starts in the UDP payload
                    int cb_data_ind  = cb_packet_ind + sizeof(cerebus_packet_header_t);
    
                    // Copy each payload entry directly to the argvPtr. dlen contains the number of 4 bytes of payload
                    for(int iChan = 0; iChan < cerebus_packet_header->dlen * 2; iChan++) {
                        memcpy(&(argvPtr[iStream][ind_samples + 1][(n[iStream] + iChan*graph_parameters.samp_per_stream[iStream]) * sizeof(int16_t)]), &udp_packet_payload[cb_data_ind + 2*iChan], sizeof(int16_t));
                    }
                    n[iStream]++;
                }
                if (n[iStream] == graph_parameters.samp_per_stream[iStream]) {

                    argvlen[iStream][ind_samples + 1] = sizeof(int16_t) * n[iStream] * graph_parameters.chan_per_stream[iStream];

                    argvlen[iStream][ind_cerebus_timestamps + 1]      = sizeof(int32_t) * n[iStream];
                    argvlen[iStream][ind_current_time + 1]            = sizeof(struct timeval) * n[iStream];
                    argvlen[iStream][ind_udp_received_time + 1]       = sizeof(struct timeval) * n[iStream];
                
                    //printf("n = %d\n", n[iStream]);
                    //for(int jj = 0; jj< n[iStream]; jj++)
                        //printf("argvPtr[%d][%d] == %s\n",iStream, jj, argvPtr[iStream][jj]);
                    /* print_neural_argv(argc, argv, argvlen); */
                    /* return 0; */

                    // Everything we've done is just to get to this one line. Whew!
                    freeReplyObject(redisCommandArgv(redis_context,  argc, (const char**) argvPtr[iStream], argvlen[iStream]));

                
                    // Since we've pushed our data to Redis, restart the data collection
                    n[iStream] = 0;
                }
            }

            // Regardless of what type of packet we got, advance to the next cerebus packet start location
            cb_packet_ind = cb_packet_ind + sizeof(cerebus_packet_header_t) + (4 * cerebus_packet_header->dlen);





        }
    }

    // We should never get here, but we should clean our data anyway.
    for (int ii = 0; ii < numStreams; ii++) {
        for (int jj = 0; jj < argc; jj++) {
            free(argvPtr[ii][jj]);
        }
    }
    free(argvlen);
    free(buffer);
    return 0;
}

//------------------------------------
// Initialization functions
//------------------------------------


void initialize_redis(char *yaml_path) {

    printf("[%s] Initializing Redis...\n", PROCESS);

    char redis_ip[16]       = {0};
    char redis_port[16]     = {0};

    load_YAML_variable_string(PROCESS, yaml_path, "redis_ip",   redis_ip,   sizeof(redis_ip));
    load_YAML_variable_string(PROCESS, yaml_path, "redis_port", redis_port, sizeof(redis_port));

    printf("[%s] From YAML, I have redis ip: %s, port: %s\n", PROCESS, redis_ip, redis_port);

    printf("[%s] Trying to connect to redis.\n", PROCESS);

    redis_context = redisConnect(redis_ip, atoi(redis_port));
    if (redis_context->err) {
        printf("[%s] Redis connection error: %s\n", PROCESS, redis_context->errstr);
        exit(1);
    }

    printf("[%s] Redis initialized.\n", PROCESS);
     
}

void initialize_signals() {

    printf("[%s] Attempting to initialize signal handlers.\n", PROCESS);

    signal(SIGINT, &handler_SIGINT);

    printf("[%s] Signal handlers installed.\n", PROCESS);
}

int initialize_socket(char *yaml_path) {

    // Create a UDP socket
   	int fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP ); 
    if (fd == 0) {
        perror("[cerebusAdapter] socket failed"); 
        exit(EXIT_FAILURE); 
    }
    int one = 1;
    
    //Set socket permissions so that we can listen to broadcasted packets
    if (setsockopt(fd,SOL_SOCKET,SO_BROADCAST, (void *) &one, sizeof(one)) < 0) {
        perror("[cerebusAdapter] socket permission failure"); 
        exit(EXIT_FAILURE); 
    }

    //Set socket permissions that we get a timestamp from when UDP was received
    if (setsockopt(fd,SOL_SOCKET,SO_TIMESTAMP , (void *) &one, sizeof(one)) < 0) {
        perror("[cerebusAdapter] timestamp failure"); 
        exit(EXIT_FAILURE); 
    }

    // Set timeout for socket, so that we can handle SIGINT cleanly
    struct timeval timeout;      
    timeout.tv_sec = 0;
    timeout.tv_usec = 100000;
    if (setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO , (char *) &timeout, sizeof(timeout)) < 0) {
        perror("[cerebusAdapter] timeout failure"); 
        exit(EXIT_FAILURE); 
    }


    char broadcast_port_string[16] = {0};
    load_YAML_variable_string(PROCESS, yaml_path, "broadcast_port", broadcast_port_string, sizeof(broadcast_port_string));
    int broadcast_port = atoi(broadcast_port_string);
    printf("[%s] I will be listening on port %d\n", PROCESS, broadcast_port);


    // Now configure the socket
    struct sockaddr_in addr;
    memset(&addr,0,sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY); //htonl(INADDR_BROADCAST);
    addr.sin_port        = htons(broadcast_port);

     if (bind(fd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        perror("[cerebusAdapter] socket binding failure\n"); 
        exit(EXIT_FAILURE); 
     }

     printf("[%s] Socket initialized.\n",PROCESS);


     return fd;
} 

void initialize_parameters(graph_parameters_t *p, char *yaml_path) {
    char broadcast_port_string[16] = {0};
    char num_streams_string[16] = {0};
    char stream_names_string[256] = {0};
    char samp_freq_string[32] = {0};
    char packet_type_string[32] = {0};
    char chan_per_stream_string[32] = {0};
    char samp_per_stream_string[32] = {0};

    // brings in the parameters -- this returns everything as a char
    load_YAML_variable_string(PROCESS, yaml_path, "broadcast_port",    broadcast_port_string,  sizeof(broadcast_port_string));
    load_YAML_variable_string(PROCESS, yaml_path, "num_streams",       num_streams_string,     sizeof(num_streams_string));
    load_YAML_variable_string(PROCESS, yaml_path, "stream_names",      stream_names_string,    sizeof(stream_names_string));
    load_YAML_variable_string(PROCESS, yaml_path, "samp_freq",         samp_freq_string,       sizeof(samp_freq_string));
    load_YAML_variable_string(PROCESS, yaml_path, "packet_type",       packet_type_string,     sizeof(packet_type_string));
    load_YAML_variable_string(PROCESS, yaml_path, "chan_per_stream",   chan_per_stream_string, sizeof(chan_per_stream_string));
    load_YAML_variable_string(PROCESS, yaml_path, "samp_per_stream",   samp_per_stream_string, sizeof(samp_per_stream_string));

    // parsing the strings into usable content
    p->num_streams =    atoi(num_streams_string);
    p->broadcast_port = atoi(broadcast_port_string);
    // now for the tricker ones -- with potentially multiple entries

    

    parameter_array_parser(1, stream_names_string, (void *) p->stream_names);
    parameter_array_parser(0, samp_freq_string, (void *) p->samp_freq);
    parameter_array_parser(0, packet_type_string, (void *) p->packet_type); 
    parameter_array_parser(0, chan_per_stream_string, (void *) p->chan_per_stream); 
    parameter_array_parser(0, samp_per_stream_string, (void *) p->samp_per_stream); 


}


// traverse an input string with CSVs and stores it into the array at *array_ind
// this takes advantage of strtok_r to keep everything threadsafe, which isn't
// defined in c99 but is in most posix implementations
void parameter_array_parser(int is_char, char *in_string, void *array_ind) {
    // initialize an int array and a char array
    int intArr[MAXSTREAMS] = {0}; // set the array to zeros
    char *charArr[MAXSTREAMS] = {0}; // can we set an array of pointers to NULL? I think so...
    
    char *token, *saveptr, *tempstr;// output and pointer to next location
    const char *delim = "[', ]"; // not going to allow any spaces in, are we? 
    int ii;

    // repeatedly run through the string while we're not getting a NULL
    // or passing beyond the length of the arrays.
    for (ii=0, tempstr = in_string; ii<MAXSTREAMS; ii++, tempstr = NULL) {
        token = strtok_r(tempstr, delim, &saveptr);
        if(token == NULL){
            break;}
        if(is_char){
            charArr[ii] = malloc(strlen(token));
            strcpy(charArr[ii], token);
            }
        else{
            intArr[ii] = atoi(token); // store if it's an int
            }
    } 

    // pass back either the character or integer array
    if(is_char){
        memcpy(array_ind, (const void *) charArr, sizeof(charArr));
    }
    else {
        memcpy(array_ind, (const void *) &intArr, sizeof(intArr));
    }


}



// Do we want the system to be realtime?  Setting the Scheduler to be real-time, priority 80
void initialize_realtime(char *yaml_path) {

    char sched_fifo_string[16] = {0};
    load_YAML_variable_string(PROCESS, yaml_path, "sched_fifo", sched_fifo_string, sizeof(sched_fifo_string));

    if (strcmp(sched_fifo_string, "True") != 0) {
        return;
    }


    printf("[%s] Setting Real-time scheduler!\n", PROCESS);
    const struct sched_param sched= {.sched_priority = 80};
    sched_setscheduler(0, SCHED_FIFO, &sched);
}

void shutdown_process() {

    printf("[%s] SIGINT received. Shutting down.\n", PROCESS);

    printf("[%s] Setting scheduler back to baseline.\n", PROCESS);
    const struct sched_param sched= {.sched_priority = 0};
    sched_setscheduler(0, SCHED_OTHER, &sched);

    printf("[%s] Shutting down redis.\n", PROCESS);

    redisFree(redis_context);

    printf("[%s] Exiting.\n", PROCESS);
    
    exit(0);
}

//------------------------------------
// Handler functions
//------------------------------------

void handler_SIGINT(int exitStatus) {
    flag_SIGINT++;
}

//------------------------------------
// Helper function
//------------------------------------

// Quick and dirty function used for debugging purposes
void print_argv(int argc, char **argv, size_t *argvlen) {
    printf("argc = %d\n", argc);

    for (int i = 0; i < 5; i++){
        printf("%02d. (%s) - [%ld]\n", i, argv[i], argvlen[i]);
    }

    for (int i = 5; i < 7; i+=2){
        printf("%02d. (%s) [%ld] - [", i, argv[i], argvlen[i+1]);

        for (int j = 0; j < argvlen[i+1]; j+= sizeof(uint32_t)) {
            printf("%u,",  (uint32_t) argv[i+1][j]);
        }

        printf("]\n");
    }

    for (int i = 7; i < 11; i+=2){
        printf("%02d. (%s) [%ld] - [", i, argv[i], argvlen[i+1]);

        for (int j = 0; j < argvlen[i+1]; j+= sizeof(struct timeval)) {
            struct timeval time;
            memcpy(&time,&argv[i+1][j], sizeof(struct timeval));
            long milliseconds = time.tv_sec * 1000 + time.tv_usec / 1000;
            long microseconds = time.tv_usec % 1000;
            printf("%ld.%03ld,", milliseconds,microseconds);
        }

        printf("]\n");
    }


    for (int i = 11; i < argc; i+=2){
        printf("%02d. (%s) [%ld] - [", i, argv[i], argvlen[i+1]);

        for (int j = 0; j < argvlen[i+1]; j+= sizeof(uint16_t)) {
            printf("%u,",  (uint16_t) argv[i+1][j]);
        }

        printf("]\n");
    }

}

