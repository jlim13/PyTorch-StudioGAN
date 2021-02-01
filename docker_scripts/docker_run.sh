#docker build -t research_img --build-arg local_xauth_="$localxauth_var" --build-arg local_display="$displayvar" .

docker run -it --runtime=nvidia --ipc=host --net=host \
       --privileged \
       --env DISPLAY \
       --volume /playpen/john/docker_scripts/.vimrc:/root/.vimrc \
       --volume /tmp/.X11-unix:/tmp/.X11-unix \
       --volume /var/run/docker.sock:/var/run/docker.sock \
       --env XAUTHORITY=/root/.Xauthority \
       -v /playpen/john/:/root/code \
       -v /data/:/root/data \
       --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
       mgkang/studio_gan:latest bash

#
