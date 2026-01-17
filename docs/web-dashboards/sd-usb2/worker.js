let leftChunks = [];
let rightChunks = [];
let recordingActive = false;

onmessage = function(e) {
    if (e.data.type === 'START_RECORDING') {
        leftChunks = [];
        rightChunks = [];
        recordingActive = true;
    } 

    if (e.data.type === 'PROCESS_CHUNK') {
        if (recordingActive) {
            leftChunks.push(new Float32Array(e.data.payload.left));
            rightChunks.push(new Float32Array(e.data.payload.right));
        }
    }
    
    if (e.data.type === 'GENERATE_WAV') {
        recordingActive = false; 
        if (leftChunks.length === 0) return;

        // 1. Stereo Blob (For Download)
        const blobStereo = exportWAV(leftChunks, rightChunks, 2, 16000);
        
        // 2. Mono Blobs (For Replay)
        const blobLeft = exportWAV(leftChunks, null, 1, 16000);
        const blobRight = exportWAV(rightChunks, null, 1, 16000);

        postMessage({ 
            type: 'WAV_RESULT', 
            stereo: blobStereo, 
            monoL: blobLeft, 
            monoR: blobRight 
        });
        
        leftChunks = [];
        rightChunks = [];
    }
};

function exportWAV(lChunks, rChunks, numChannels, sampleRate) {
    const totalLen = lChunks.reduce((acc, c) => acc + c.length, 0);
    const buffer = new ArrayBuffer(44 + totalLen * numChannels * 2);
    const view = new DataView(buffer);

    // --- WRITE HEADER ---
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + totalLen * numChannels * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); 
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * 2, true); 
    view.setUint16(32, numChannels * 2, true); 
    view.setUint16(34, 16, true); 
    writeString(view, 36, 'data');
    view.setUint32(40, totalLen * numChannels * 2, true);

    // --- WRITE DATA ---
    let offset = 44;
    
    const writeSample = (val) => {
        let s = Math.max(-1, Math.min(1, val));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
    };

    if (numChannels === 1) {
        for (let i = 0; i < lChunks.length; i++) {
            const ch = lChunks[i];
            for (let j = 0; j < ch.length; j++) writeSample(ch[j]);
        }
    } else {
        for (let i = 0; i < lChunks.length; i++) {
            const l = lChunks[i];
            const r = rChunks[i];
            for (let j = 0; j < l.length; j++) {
                writeSample(l[j]);
                writeSample(r[j]);
            }
        }
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}