#docker build -t research_img --build-arg local_xauth_="$localxauth_var" --build-arg local_display="$displayvar" .


docker run -it --gpus all \
        --shm-size 256g -p 6006:6006 \
        -v /playpen/john/:/root/code \
        -v /data/:/root/data \
        --volume /playpen/john/docker_scripts/.vimrc:/root/.vimrc \
        --workdir /root/code \
        mgkang/studio_gan:latest /bin/zsh

