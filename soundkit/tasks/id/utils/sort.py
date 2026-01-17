from collections import defaultdict
import os
import yaml

def grouping_spks_sentences(
        fnames: list,
        path: str) -> None:
    """
    [
    .../spk-xxxx/.../sentence-xxxx-xxxx-xxxx.tfrecord,
    .../spk-xxxx/.../sentence-xxxx-xxxx-xxxx.tfrecord,
    .../spk-xxxx/.../sentence-xxxx-xxxx-xxxx.tfrecord,
    ]
    
    This function groups
    [
        [ # speaker_A
            [sentence1_noise1, sentence1_noise2],
            [sentence2_noise1, sentence2_noise2],
        ]
        [ # speaker_B
            [sentence1_noise1, sentence1_noise2],
            [sentence2_noise1, sentence2_noise2],
        ]
    ]
    """
    # Step 1: Group by speaker → then group by basename
    speaker_map = defaultdict(lambda: defaultdict(list))  # {spk: {basename: [full_paths]}}

    for fname in fnames:
        parts = fname.split('/')
        spk_id = parts[4]  # 'spk-XXXX'
        basename = os.path.basename(fname)  # e.g. '511-131228-0073.tfrecord'
        speaker_map[spk_id][basename].append(fname)

    # Step 2: Build final structure
    final_output = []

    for spk, file_group in speaker_map.items():
        grouped_files = []
        # Sort by numeric basename keys
        for base in sorted(file_group, key=lambda x: [int(p) for p in x.replace('.tfrecord', '').split('-')]):
            grouped_files.append(file_group[base])  # All paths with the same basename
        final_output.append(grouped_files)

    # Step 3: Save as YAML
    with open(path, 'w') as yaml_file:
        yaml.dump(final_output, yaml_file, default_flow_style=False, allow_unicode=True)
