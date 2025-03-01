NVCC = nvcc
SRC = main.cpp *.cu
EXE = main

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) -o $(EXE) $(SRC)

clean:
	rm -f $(EXE)
