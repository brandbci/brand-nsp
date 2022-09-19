PROJECT=cerebusAdapter

ROOT ?= ../..
include $(ROOT)/setenv.mk

CC = gcc
DEFAULTFLAGS = -Wall 
CFLAGS = -O2 $(DEFAULTFLAGS)
CFLAGSUNOPT = -O0 $(DEFAULTFLAGS)
CFLAGS+ = -g -O2 $(DEFAULTFLAGS)

LINKS= -L$(HIREDIS_PATH)
LDPATH = -Wl,-rpath=$(HIREDIS_PATH)
LDFLAGS = -lhiredis

CINCLUDE = -I$(ROOT)/lib/c_code/brands/ \
		   -I$(HIREDIS_PATH)


CFILES = $(ROOT)/lib/c_code/brands/tools_CParser.c

//TARGET=$(BIN_PATH)/$(PROJECT)
TARGET=$(PROJECT).bin

all: cerebusAdapter

cerebusAdapter: cerebusAdapter.c
	$(CC) $(CFLAGS+) $(CINCLUDE) $(LINKS) cerebusAdapter.c $(CFILES) $(LDPATH) -o $(TARGET) $(LDFLAGS)

clean:
	$(RM) $(TARGET)
