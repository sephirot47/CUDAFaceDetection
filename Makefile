all: image_load.exe 

image_load.o: image_load.cpp
	g++ -std=c++11 -c image_load.cpp -o image_load.o

stbi.o: stbi.cpp
	g++ -std=c++11 -c stbi.cpp -o stbi.o

#stbi_write.o: stbi_write.h
#	g++ -std=c++11 -c stbi_write.h -o stbi_write.o
	
image_load.exe: image_load.o stbi.o #stbi_write.o
	g++ -std=c++11 image_load.o stbi.o -o image_load.exe

clean:
	rm -rf *.o *.exe
