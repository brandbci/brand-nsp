#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h> 
#include <sys/socket.h>
#include <arpa/inet.h>
#include "hiredis.h"
#include "brand.h"

#define MILLION 1000000.0
#define BILLION 1000000000.0

#define MAXSTREAMS 10 // maximum number of streams allowed -- to set up all of the arrays as necessary


// Cerebus packet definition, adapted from the standard Blackrock library
// https://github.com/neurosuite/libcbsdk/cbhwlib/cbhwlib.h
typedef struct cerebus_packet_header_t {
    uint64_t time;      // 8 bytes
    uint16_t chid;      // 2 bytes
    uint16_t type;      // 2 bytes
    uint16_t dlen;      // 2 bytes
    char reserved[2];   // 2 bytes
} cerebus_packet_header_t;


// this allows for only a max streams based on constant above
// change that constant if you want
typedef struct graph_parameters_t {
    char broadcast_ip[20];
    int broadcast_port;
    char broadcast_device[20];
    int num_streams;
    char **stream_names;
    int *samp_freq;
    int *packet_type;
    int *chan_per_stream;
    int *samp_per_stream;
} graph_parameters_t;


// intialize support functions
void initialize_signals();
int  initialize_socket(char *broadcast_ip, int broadcast_port, char *broadcast_device);
void initialize_parameters(redisContext *c, graph_parameters_t *p);
void handler_SIGINT(int exitStatus);
void shutdown_process();
void print_argv(int, char **, size_t *);
uint32_t parse_ip_str(char *ip_str);
double diff_timespec(const struct timespec *time1, const struct timespec *time0);

char NICKNAME[] = "cerebusAdapter";
char SUPERGRAPH_ID[256];

int flag_SIGINT = 0;



int main (int argc_main, char **argv_main) {

    initialize_signals();

    redisContext *c = parse_command_line_args_init_redis(argc_main, argv_main, NICKNAME);  
    emit_status(c, NICKNAME, NODE_STARTED, NULL);

    //yaml_parameters_t yaml_parameters = {0};
    graph_parameters_t graph_parameters = {};
    initialize_parameters(c, &graph_parameters); // this is a little more complicated with a var num IOs
    // graph_parameters_t graph_parameters = (graph_parameters_t) {51002, 3, {"cb_gen_1"}, {30000}, {6}, {96}, {30}};
    int numStreams = graph_parameters.num_streams; //will use this a lot, so pull it out

    int udp_fd = initialize_socket(graph_parameters.broadcast_ip, graph_parameters.broadcast_port,
        graph_parameters.broadcast_device);


    // argc    : The number of arguments in argv. The calculation is:
    //           int argc = 3 + 2 * 6;
    //           3                  -> xadd cerebusAdapter *
    //           6                  -> timestamps (3 types), tracking id, write_timestamp, and sample array
    //           2                  -> (key, value) pairs
    // argvlen : The length of the strings in each argument of argv
    //           This will be an array of pointers since we've got a 
    //           variable number of streams
    //
    // We will be using the same argc and indices for all different frequencies,
    // but we will use different argv and argvlen for each.
    int argc        = 3 + (2 * 6); // argcount = xadd + key:value for everything else

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
    int ind_current_timestamps          = ind_cerebus_timestamps + 2;   // current_time [data]
    int ind_udp_recv_timestamps         = ind_current_timestamps + 2;      	// udp_received_time [data]
    int ind_tracking_id                 = ind_udp_recv_timestamps + 2;    // tracking_id [data]
    int ind_write_timestamp             = ind_tracking_id + 2;          // write_timestamp [data]
    int ind_samples                     = ind_write_timestamp + 2;      // samples [data array] 
    
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
        for (int jj = 0; jj < 3; jj++) {
            argvPtr[ii][jj] = malloc(len);
        }
        
        // allocating memory for cerebus timestamps [data]
        argvPtr[ii][ind_cerebus_timestamps]             = malloc(len);
        argvPtr[ii][ind_cerebus_timestamps + 1]         = malloc(sizeof(uint64_t) * samp_per_stream);
        
        // allocating memory for current_time [data]
        argvPtr[ii][ind_current_timestamps]                   = malloc(len);
        argvPtr[ii][ind_current_timestamps + 1]               = malloc(sizeof(double) * samp_per_stream);

        // allocating memory for udp_received_time [data]
        argvPtr[ii][ind_udp_recv_timestamps]              = malloc(len);
        argvPtr[ii][ind_udp_recv_timestamps + 1]          = malloc(sizeof(double) * samp_per_stream);

        // allocating memory for tracking_id [data]
        argvPtr[ii][ind_tracking_id]                    = malloc(len);
        argvPtr[ii][ind_tracking_id + 1]                = malloc(sizeof(unsigned long));

        // allocating memory for write_timestamp [data]
        argvPtr[ii][ind_write_timestamp]                    = malloc(len);
        argvPtr[ii][ind_write_timestamp + 1]                = malloc(sizeof(double));
   
        // allocating memory for samples:  [data0 ... dataX]
        argvPtr[ii][ind_samples]                        = malloc(len);
        argvPtr[ii][ind_samples + 1]                    = malloc(sizeof(int16_t) * samp_per_stream * chan_per_stream);
 
        // At this point we start populating neural_argv strings
        // Start by adding xadd cerebusAdapter *
        // And then add the keys for num_samples, timestamps, channel list, and sample array

        argvlen[ii][0] = sprintf(argvPtr[ii][0], "%s", "xadd");
        argvlen[ii][1] = sprintf(argvPtr[ii][1], "%s", graph_parameters.stream_names[ii]);
        argvlen[ii][2] = sprintf(argvPtr[ii][2], "%s", "*");
        
        argvlen[ii][ind_cerebus_timestamps]      = sprintf(argvPtr[ii][ind_cerebus_timestamps], "%s", "timestamps");
        argvlen[ii][ind_current_timestamps]      = sprintf(argvPtr[ii][ind_current_timestamps], "%s", "BRANDS_time");
        argvlen[ii][ind_udp_recv_timestamps]     = sprintf(argvPtr[ii][ind_udp_recv_timestamps], "%s", "udp_recv_time");
        argvlen[ii][ind_tracking_id]             = sprintf(argvPtr[ii][ind_tracking_id], "%s", "tracking_id");
        argvlen[ii][ind_write_timestamp]         = sprintf(argvPtr[ii][ind_write_timestamp], "%s", "write_timestamp");
        argvlen[ii][ind_samples]                 = sprintf(argvPtr[ii][ind_samples], "%s", "samples");

    }

    printf("[%s] Entering loop...\n", NICKNAME);
    
    // How many samples have we copied 
    int n[numStreams];  
    for(int ii = 0; ii < numStreams; ii++)
        n[ii] = 0;

    // We use rcvmsg because we want to know when the kernel received the UDP packet
    // and because we want the socket read to timeout, allowing us to gracefully
    // shutdown with a SIGINT call. Using recvmsg means there's a lot more overhead
    // in actually getting to business, as seen below

    char *buffer = malloc(65535); // max size of conceivable packet
    char msg_control_buffer[65535] = {0};
    
    struct iovec message_iovec = {0};
    message_iovec.iov_base = buffer;
    message_iovec.iov_len  = 65535;

    struct msghdr message_header = {0};
    message_header.msg_name       = NULL;
    message_header.msg_iov        = &message_iovec;
    message_header.msg_iovlen     = 1;
    message_header.msg_control    = msg_control_buffer;
    message_header.msg_controllen = 65535;

    // tracking id (incremental so as to be unique per redis write). nodes take the
    // tracking_id of the entry in their input stream and put it in the entry in their
    // output stream so we can track calculations through the system.
    unsigned long tracking_id = 0;

    struct timespec spike_err_time;
    clock_gettime(CLOCK_MONOTONIC, &spike_err_time);
    uint32_t num_spike_pkts = 0;

    double current_timedstamp;
    struct timespec current_time_spec;
    clock_gettime(CLOCK_MONOTONIC, &current_time_spec);

    double udp_recv_timedstamp;
    struct timeval udp_recv_time_val;
    
    struct cmsghdr *cmsg_header; // Used for getting the time UDP packet was received

    // timestamp of each write to redis (and temp vars for unpacking from clock_gettime)
    double write_timestamp;
    struct timespec write_timestamp_spec;

    while (1) {
        if (flag_SIGINT) 
            shutdown_process(&graph_parameters);

        // We use recvmsg because we want to know when the kernel received the UDP packet
        int udp_packet_size = recvmsg(udp_fd, &message_header, 0);
        // The timer has timed out or there was an error with the recvmsg() call
        if (udp_packet_size  <= 0) {
            printf("[%s] timer has timed out or there was an error with the recvmsg() call!\n",NICKNAME);
            continue;
        }

        // For convenience, makes it much easier to reason about
        char *udp_packet_payload = (char*) message_header.msg_iov->iov_base;
        
        // These two lines of code get the time the UDP packet was received
        cmsg_header = CMSG_FIRSTHDR(&message_header); 
        memcpy(&udp_recv_time_val, CMSG_DATA(cmsg_header), sizeof(struct timeval));
        udp_recv_timedstamp = (double)udp_recv_time_val.tv_sec + (double)udp_recv_time_val.tv_usec / MILLION;

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

                // check that we are not getting spike packets, dlen is 0 for Cerebus heartbeat
                if ((cerebus_packet_header->type == (uint8_t) 0) && (cerebus_packet_header->dlen > (uint8_t) 0)) {
                    num_spike_pkts++;

                    // write warnings once a second
                    if (diff_timespec(&current_time_spec, &spike_err_time) >= 1.0) {
                        printf("[%s] received %d spike packets!\n",NICKNAME,num_spike_pkts);
                        memcpy(&spike_err_time, &current_time_spec, sizeof(struct timespec));
                    }
                }

                if (cerebus_packet_header->type == graph_parameters.packet_type[iStream]) {
                    
                    // This gets the current system time
                    clock_gettime(CLOCK_MONOTONIC, &current_time_spec);
                    current_timedstamp = (double)current_time_spec.tv_sec + (double)current_time_spec.tv_nsec / BILLION;
                    
                    // Copy the timestamp information into argvPtr
                    memcpy( &argvPtr[iStream][ind_cerebus_timestamps + 1][n[iStream] * sizeof(uint64_t)],    &cerebus_packet_header->time,      sizeof(uint64_t));
                    // memcpy( &argvPtr[iStream][ind_current_timestamps + 1][n[iStream] * sizeof(double)],      &current_timedstamp,               sizeof(double));
                    // memcpy( &argvPtr[iStream][ind_udp_recv_timestamps + 1][n[iStream] * sizeof(double)],     &udp_recv_timedstamp,              sizeof(double));
                    memcpy( &argvPtr[iStream][ind_current_timestamps + 1][n[iStream] * sizeof(struct timeval)],      &current_time_spec,               sizeof(struct timeval));
                    memcpy( &argvPtr[iStream][ind_udp_recv_timestamps + 1][n[iStream] * sizeof(struct timeval)],     &udp_recv_time_val,              sizeof(struct timeval));

                    // The index where the data starts in the UDP payload
                    int cb_data_ind  = cb_packet_ind + sizeof(cerebus_packet_header_t);
    
                    // Copy each payload entry directly to the argvPtr. dlen contains the number of 4 bytes of payload
                    // for(int iChan = 0; iChan < cerebus_packet_header->dlen * 2; iChan++) {
                    //     memcpy(&(argvPtr[iStream][ind_samples + 1][(n[iStream] + iChan*graph_parameters.samp_per_stream[iStream]) * sizeof(int16_t)]), &udp_packet_payload[cb_data_ind + 2*iChan], sizeof(int16_t));
                    // }
                    memcpy(&(argvPtr[iStream][ind_samples + 1][n[iStream] * cerebus_packet_header->dlen * 4]), &udp_packet_payload[cb_data_ind], cerebus_packet_header->dlen * 4);
                    n[iStream]++;
                }

                if (n[iStream] == graph_parameters.samp_per_stream[iStream]) {
                    // Update the length (in bytes) of each data field
                    argvlen[iStream][ind_cerebus_timestamps + 1]      = sizeof(uint64_t) * n[iStream];
                    argvlen[iStream][ind_current_timestamps + 1]      = sizeof(struct timeval) * n[iStream];
                    argvlen[iStream][ind_udp_recv_timestamps + 1]     = sizeof(struct timeval) * n[iStream];
                    argvlen[iStream][ind_tracking_id + 1]             = sizeof(unsigned long);
                    argvlen[iStream][ind_write_timestamp + 1]         = sizeof(struct timeval);
                    argvlen[iStream][ind_samples + 1]                 = sizeof(int16_t) * n[iStream] * graph_parameters.chan_per_stream[iStream];

                    // Use a unique tracking_id for data to flow through the graph with.
                    tracking_id++;
                    memcpy( &argvPtr[iStream][ind_tracking_id + 1][0], &tracking_id, sizeof(unsigned long));

                    // Get the system time immediately before writing to redis.
                    clock_gettime(CLOCK_MONOTONIC, &write_timestamp_spec);
                    // TODO: why does the division lose precision? I'd have thought the numbers are within range.
                    write_timestamp = (double)write_timestamp_spec.tv_sec + (double)write_timestamp_spec.tv_nsec / BILLION;
                    memcpy( &argvPtr[iStream][ind_write_timestamp + 1][0], &write_timestamp_spec, sizeof(struct timeval));

                    // Everything we've done is just to get to this one line. Whew!
                    freeReplyObject(redisCommandArgv(c,  argc, (const char**) argvPtr[iStream], argvlen[iStream]));

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

void initialize_signals() {

    printf("[%s] Attempting to initialize signal handlers.\n", NICKNAME);

    signal(SIGINT, &handler_SIGINT);

    printf("[%s] Signal handlers installed.\n", NICKNAME);
}

int initialize_socket(char *broadcast_ip, int broadcast_port, char *broadcast_device) {

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
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;
    if (setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO , (char *) &timeout, sizeof(timeout)) < 0) {
        perror("[cerebusAdapter] timeout failure"); 
        exit(EXIT_FAILURE); 
    }

    // Load the broadcast IP
    char broadcast_ip_string[INET_ADDRSTRLEN];
    strcpy(broadcast_ip_string, broadcast_ip);
    uint32_t broadcast_ip_int = parse_ip_str(broadcast_ip_string);
    printf("[%s] Listening on port %d\n", NICKNAME, broadcast_port);

    // Bind socket to a specific interface
    char interface[20] = {0};
    strcpy(interface, broadcast_device);
    printf("[%s] Binding socket to device: %s\n", NICKNAME, interface);

    int so = setsockopt(fd, SOL_SOCKET, SO_BINDTODEVICE, &interface,
        sizeof(interface));
    if (so < 0) {
        perror("[cerebusAdapter] interface binding error");
    }

    // Now configure the socket
    struct sockaddr_in addr;
    memset(&addr,0,sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = broadcast_ip_int;
    addr.sin_port        = htons(broadcast_port);

     if (bind(fd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        perror("[cerebusAdapter] Socket binding failure");  
        exit(EXIT_FAILURE); 
     }

     printf("[%s] Socket initialized.\n",NICKNAME);

     return fd;
} 

void initialize_parameters(redisContext *c, graph_parameters_t *p)
{
    // Initialize Supergraph_ID 
    SUPERGRAPH_ID[0] = '0';
    // Now fetch data from the supergraph and populate entries
    redisReply *reply = NULL;
    const nx_json *supergraph_json = get_supergraph_json(c, reply, SUPERGRAPH_ID); 
    if (supergraph_json == NULL) {
        emit_status(c, NICKNAME, NODE_FATAL_ERROR,
                    "No supergraph found for initialization. Aborting.");
        exit(1);
    }

    strcpy(p->broadcast_ip, get_parameter_string(supergraph_json, NICKNAME , "broadcast_ip"));
    p->broadcast_port = get_parameter_int(supergraph_json, NICKNAME , "broadcast_port");
    strcpy(p->broadcast_device, get_parameter_string(supergraph_json, NICKNAME , "broadcast_device"));
    get_parameter_list_string(supergraph_json, NICKNAME, "stream_names", &p->stream_names, &p->num_streams);
    get_parameter_list_int(supergraph_json, NICKNAME, "samp_freq", &p->samp_freq, &p->num_streams);
    get_parameter_list_int(supergraph_json, NICKNAME, "packet_type", &p->packet_type, &p->num_streams);
    get_parameter_list_int(supergraph_json, NICKNAME, "chan_per_stream", &p->chan_per_stream, &p->num_streams);
    get_parameter_list_int(supergraph_json, NICKNAME, "samp_per_stream", &p->samp_per_stream, &p->num_streams);

    printf("[%s] Initialization complete. Reading from %d streams.\n", NICKNAME, p->num_streams);

    // Free memory, since all relevant information has been transffered to the parameter struct at this point
    // using memcpy and strcpy commands
    nx_json_free(supergraph_json);
    freeReplyObject(reply);
}


void shutdown_process(graph_parameters_t *p) {
    printf("[%s] SIGINT received. Exiting.\n", NICKNAME);
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

uint32_t parse_ip_str(char *ip_str) {
    unsigned char buf[sizeof(struct in_addr)];
    int domain, s;

    domain = AF_INET;
    s = inet_pton(domain, ip_str, buf);
    if (s <= 0) {
        if (s == 0) {
            fprintf(stderr, "[%s] Invalid IP address: '%s'\n", NICKNAME, ip_str);
        } else {
            perror("inet_pton");
        }
        fprintf(stderr, "[%s] Reverting to 255.255.255.255\n", NICKNAME);
        s = inet_pton(domain, "255.255.255.255", buf);
    }

    uint32_t ip_num;
    memcpy(&ip_num, buf, 4);

    return ip_num;
}

double diff_timespec(const struct timespec *time1, const struct timespec *time0) {
  return (time1->tv_sec - time0->tv_sec)
      + (time1->tv_nsec - time0->tv_nsec) / 1000000000.0;
}