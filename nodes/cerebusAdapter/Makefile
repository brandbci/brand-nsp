PROJECT=cerebusAdapter

ifneq ($(CONDA_DEFAULT_ENV),rt)
$(error real-time conda env (rt) not active)
endif

ROOT ?= ../..
include $(ROOT)/setenv.mk

CC = gcc
DEFAULTFLAGS = -Wall 
CFLAGS = -O2 $(DEFAULTFLAGS)
CFLAGSUNOPT = -O0 $(DEFAULTFLAGS)
CFLAGS+ = -g -O2 $(DEFAULTFLAGS)

LINKS= -L$(HIREDIS_PATH) -L$(ROOT)/lib/nxjson/nxjson.h
LDPATH = -Wl,-rpath=$(HIREDIS_PATH)
LDFLAGS = -lhiredis

CINCLUDE = -I$(ROOT)/lib/c/brand/ \
		   -I$(HIREDIS_PATH) \
		   -I$(ROOT)/lib/nxjson


CFILES = $(ROOT)/lib/c/brand/brand.c \
		 $(ROOT)/lib/nxjson/nxjson.c


TARGET=$(PROJECT).bin

all: cerebusAdapter

cerebusAdapter: cerebusAdapter.c
	$(CC) $(CFLAGS+) $(CINCLUDE) $(LINKS) cerebusAdapter.c $(CFILES) $(LDPATH) -o $(TARGET) $(LDFLAGS)

clean:
	$(RM) $(TARGET)
