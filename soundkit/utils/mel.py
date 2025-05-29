"""Generate Mel frequency banks. We use the original source code from Haytham M. Fayek.
"https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html"

 We simply modify the coefficients of Mel freq banks to fake fixed-point.
"""
import numpy as np
import matplotlib.pyplot as plt
from soundkit.utils.converter_fix_point import fakefix

def gen_mel_bank(
                fftsize         = 512,
                nfilt           = 40,
                sample_rate     = 16000,
                mel_c_path    = None):
    """
    Generate mel bank coefficients
    """
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin_mel = np.floor((fftsize + 1) * hz_points / sample_rate)
    # print(bin_mel)
    # print(len(bin_mel))
    fbank = np.zeros((nfilt, int(np.floor(fftsize / 2 + 1))))
    if mel_c_path is not None:
        filename = f'{mel_c_path}/melSpec_coeff_nfilt.c'
        file = open(filename, 'w') # pylint: disable=unspecified-encoding
        file.write('#include <stdint.h>\n')
        file.write('#include "ambiq_nnsp_const.h"\n')
        file.write(f'const int16_t num_mfltrBank_nfilt{nfilt} = {nfilt};\n')
        file.write(f'//const int16_t num_fft = {fftsize};\n' )
        file.write(f"const int16_t mfltrBank_coeff_nfilt{nfilt}[]={{\n")

    for idx in range(1, nfilt + 1):
        f_m_minus = int(bin_mel[idx - 1])   # left
        f_m = int(bin_mel[idx])             # center
        f_m_plus = int(bin_mel[idx + 1])    # right
        # print(f"left: {f_m_minus+1}, right: {f_m_plus-1}")
        for k in range(f_m_minus, f_m):
            fbank[idx - 1, k] = (k - bin_mel[idx - 1]) / (bin_mel[idx] - bin_mel[idx - 1])
        for k in range(f_m, f_m_plus):
            fbank[idx - 1, k] = (bin_mel[idx + 1] - k) / (bin_mel[idx + 1] - bin_mel[idx])

        if mel_c_path is not None:
            file.write(f'0x{int(f_m_minus+1):04X}, 0x{int(f_m_plus-1):04X},')
            for k in range(f_m_minus+1,f_m_plus):
                tmp = fbank[idx-1, k]
                val = fakefix(tmp, 16, 15)
                val = val * (2**15)
                file.write(f'0x{int(val):04X},' )
            file.write(f'// {idx-1}\n' )
    fbank_q = fakefix(fbank, 16, 15)
    if mel_c_path is not None:
        file.write('};\n')
        file.close()
    return fbank_q

def float2fix(data_in, nfrac, bitwidth):
    """
    Floating point to int
    """
    max_val = 2**(bitwidth-1) - 1
    min_val = -2**(bitwidth-1)

    out = np.minimum(np.maximum(np.floor(data_in * 2**nfrac), min_val), max_val).astype(int)

    return out

def gen_mel_c(file_path, bank_name, mel_filters, bank_type='mel'):
    with open(f"{file_path}", "w") as file:
        file.write(f"// bank type: {bank_type}\n")
        file.write("#include <stdint.h>\n")
        file.write(f"const int16_t mel_coeff_nfilt{len(mel_filters)} = {len(mel_filters)};\n")
        file.write(f"const int16_t {bank_name}[] = {{\n")
        for i_m, vec in enumerate(mel_filters):
            indices=np.array([],dtype=int)  # Initialize an empty array to store indices
            values=np.array([], dtype=np.float32)  # Initialize an empty array to store values
            for i, v in enumerate(vec):
                if v > 0:
                    indices = np.append(indices, i)
                    values = np.append(values, v)

            M = indices.max()
            m = indices.min()
            file.write(f"0x{m:04X}, 0x{M:04X}, ")  # Print min and max indices
            for v in values:
                # Convert float value to fixed-point representation
                val = float2fix(v, 15, 16)
                file.write(f"0x{val:04X}, ")  # Print fixed-point value
            file.write(f"// mel_{i_m}: start_bin, end_bin, values")
            file.write("\n")
        file.write(f"}};\n")

if __name__ == '__main__':
    fbanks = gen_mel_bank(  fftsize         = 512,
                            nfilt           = 72,
                            sample_rate     = 16000,
                            mel_c_path      = '.')
    gen_mel_c('./mel.c', fbanks)
    
    import pdb; pdb.set_trace()
    fig = plt.figure()
    for i, bank in enumerate(fbanks):
        plt.plot(bank)
    plt.show()
