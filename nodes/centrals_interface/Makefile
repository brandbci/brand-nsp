PROJECT=centrals_interface

ifneq ($(CONDA_DEFAULT_ENV),rt)
$(error real-time conda env (rt) not active)
endif

ROOT ?=../..
include $(ROOT)/setenv.mk

PYTHON_VERSION=3.8 # This works for rt env
PYTHON_LIB=python$(PYTHON_VERSION)

LIBPYTHON=$(CONDA_PREFIX)/lib/
INCPYTHON=$(CONDA_PREFIX)/include/$(PYTHON_LIB)

TARGET=$(PROJECT).bin
CYTHON_TARGET=$(PROJECT).c

all:
	cp $(PROJECT).py $(PROJECT).pyx
	cython --embed $(PROJECT).pyx -o $(CYTHON_TARGET)
	gcc $(CYTHON_TARGET) -o $(TARGET) -I $(INCPYTHON) -L $(LIBPYTHON)  -Wl,-rpath=$(LIBPYTHON) -l$(PYTHON_LIB) -lpthread -lm -lutil -ldl
	$(RM) $(PROJECT).pyx $(CYTHON_TARGET)
clean:
	$(RM) $(PROJECT).pyx $(CYTHON_TARGET)
