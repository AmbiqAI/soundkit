# for prob in 0.2 0.4 0.6; do
#     echo "=== Running with data.reverb_prob=${prob} ==="

#     soundkit -m data  -c configs/se.yaml data.reverb_prob=${prob}
#     soundkit -m train -c configs/se.yaml data.reverb_prob=${prob}

#     # soundkit -m evaluate -c configs/se.yaml data.reverb_prob=${prob}

# done

soundkit -t vad -m data  -c configs/vad/vad_16k.yaml
soundkit -t vad -m train -c configs/vad/vad_16k.yaml

# soundkit -t vad -m data  -c configs/vad/vad_8k.yaml
# soundkit -t vad -m train -c configs/vad/vad_8k.yaml