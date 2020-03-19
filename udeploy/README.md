## uDeploy example command

```
ma train tf docker --zone phx4-prod02 --respool /UberAI \
                   --num-cpus 8 --memory-size-mb 32768 \
                   --custom-docker ${CGP_DOCKER_IMAGE} \
                   --command-line '<full command>'
```
