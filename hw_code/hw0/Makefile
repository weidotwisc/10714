# Determine the operating system
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin) # macOS
    CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes) -undefined dynamic_lookup
else # Linux
    CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes)
endif

# NOTE: on MacOS you need to add an addition flag: -undefined dynamic_lookup
default:
	c++  $(CXXFLAGS) src/simple_ml_ext.cpp -o src/simple_ml_ext.so

clean:
	$(RM) -fr src/simple_ml_ext.so

