all: image_load.exe 

image_load.o: image_load.cpp
	g++ -std=c++11 -c image_load.cpp -o image_load.o

stbi.o: stbi.cpp
	g++ -std=c++11 -c stbi.cpp -o stbi.o
	
image_load.exe: image_load.o stbi.o
	g++ -std=c++11 image_load.o stbi.o -o image_load.exe

clean:
	rm -rf *.o *.exe
