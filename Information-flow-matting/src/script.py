import os 
import numpy as np 
import time 

def loop():
	dira = "../../data/trimap_lowres/Trimap1"
	dirb = "../../data/trimap_lowres/Trimap2"
	dirc = "../../data/trimap_lowres/Trimap3"
	dir1 = "../../data/input_lowres"

	# dir1 = "out2/Trimap"
	l = os.listdir(dir1)
	start_time = time.time()
	
	for file in l:
		f = str(file)
		# f = "troll.png"
		img_path = dir1+'/'+f
	
		start_time = time.time()
		save_path = "out3/Trimap1/"+f
		tri_map = dira+'/'+f
		# main(img_path, tri_map, save_path)
		os.system("g++ -std=c++11 alphac.cpp `pkg-config --cflags --libs opencv`")
		os.system("./a.out "+img_path+" "+tri_map+" "+save_path)
		print("--- %s seconds ---" % (time.time() - start_time))


		'''
		start_time = time.time()
		save_path = "out3/Trimap2/"+f
		tri_map = dirb+'/'+f
		main(img_path, tri_map, save_path)
		os.system("g++ -std=c++11 alphac.cpp `pkg-config --cflags --libs opencv`")
		os.system("./a.out "+img_path+" "+tri_map+" "+save_path)
		print("--- %s seconds ---" % (time.time() - start_time))
		# exit()

		start_time = time.time()
		save_path = "out3/Trimap3/"+f
		tri_map = dirc+'/'+f
		main(img_path, tri_map, save_path)
		os.system("g++ -std=c++11 alphac.cpp `pkg-config --cflags --libs opencv`")
		os.system("./a.out "+img_path+" "+tri_map+" "+save_path)
		print("--- %s seconds ---" % (time.time() - start_time))
		start_time = time.time()
		
		'''
	

if __name__ == "__main__":
	loop()