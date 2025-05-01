./shaderdump.py -i examples/zippyzaps.frag -o frames/frames_%04d.jpg -t0 0.0 -t1 4. -fps 60
ffmpeg -framerate 60 -i frames/frames_%04d.jpg -c:v libx264 -pix_fmt yuv420p output.mp4
