#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <regex>

extern "C" {

// Callback for progress reporting
typedef void (*whisper_quantize_progress_callback)(float progress);

// Quantization result codes
enum whisper_quantize_result {
    WHISPER_QUANTIZE_OK = 0,
    WHISPER_QUANTIZE_ERROR_INVALID_MODEL = -1,
    WHISPER_QUANTIZE_ERROR_FILE_OPEN = -2,
    WHISPER_QUANTIZE_ERROR_FILE_WRITE = -3,
    WHISPER_QUANTIZE_ERROR_INVALID_FTYPE = -4,
    WHISPER_QUANTIZE_ERROR_QUANTIZATION_FAILED = -5,
};

// Whisper model header structure
struct whisper_model_hparams {
    int32_t n_vocab;
    int32_t n_audio_ctx;
    int32_t n_audio_state;
    int32_t n_audio_head;
    int32_t n_audio_layer;
    int32_t n_text_ctx;
    int32_t n_text_state;
    int32_t n_text_head;
    int32_t n_text_layer;
    int32_t n_mels;
    int32_t ftype;
};

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;
    std::vector<float> data;
};

// Internal quantization function based on whisper.cpp's quantize.cpp
static int whisper_model_quantize_internal(
    const std::string & fname_inp,
    const std::string & fname_out,
    ggml_ftype ftype,
    whisper_quantize_progress_callback progress_callback) {

    printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

    auto finp = std::ifstream(fname_inp, std::ios::binary);
    if (!finp) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, fname_inp.c_str());
        return WHISPER_QUANTIZE_ERROR_FILE_OPEN;
    }

    auto fout = std::ofstream(fname_out, std::ios::binary);
    if (!fout) {
        fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname_out.c_str());
        return WHISPER_QUANTIZE_ERROR_FILE_WRITE;
    }

    // Verify magic
    {
        uint32_t magic;
        finp.read((char *) &magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname_inp.c_str());
            return WHISPER_QUANTIZE_ERROR_INVALID_MODEL;
        }

        fout.write((char *) &magic, sizeof(magic));
    }

    whisper_model_hparams hparams = {};

    // Load hparams
    {
        finp.read((char *) &hparams.n_vocab,       sizeof(hparams.n_vocab));
        finp.read((char *) &hparams.n_audio_ctx,   sizeof(hparams.n_audio_ctx));
        finp.read((char *) &hparams.n_audio_state, sizeof(hparams.n_audio_state));
        finp.read((char *) &hparams.n_audio_head,  sizeof(hparams.n_audio_head));
        finp.read((char *) &hparams.n_audio_layer, sizeof(hparams.n_audio_layer));
        finp.read((char *) &hparams.n_text_ctx,    sizeof(hparams.n_text_ctx));
        finp.read((char *) &hparams.n_text_state,  sizeof(hparams.n_text_state));
        finp.read((char *) &hparams.n_text_head,   sizeof(hparams.n_text_head));
        finp.read((char *) &hparams.n_text_layer,  sizeof(hparams.n_text_layer));
        finp.read((char *) &hparams.n_mels,        sizeof(hparams.n_mels));
        finp.read((char *) &hparams.ftype,         sizeof(hparams.ftype));

        const int32_t qntvr_src = hparams.ftype / GGML_QNT_VERSION_FACTOR;
        const int32_t ftype_dst = GGML_QNT_VERSION * GGML_QNT_VERSION_FACTOR + ftype;

        fprintf(stderr, "%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
        fprintf(stderr, "%s: n_audio_ctx   = %d\n", __func__, hparams.n_audio_ctx);
        fprintf(stderr, "%s: n_audio_state = %d\n", __func__, hparams.n_audio_state);
        fprintf(stderr, "%s: n_audio_head  = %d\n", __func__, hparams.n_audio_head);
        fprintf(stderr, "%s: n_audio_layer = %d\n", __func__, hparams.n_audio_layer);
        fprintf(stderr, "%s: n_text_ctx    = %d\n", __func__, hparams.n_text_ctx);
        fprintf(stderr, "%s: n_text_state  = %d\n", __func__, hparams.n_text_state);
        fprintf(stderr, "%s: n_text_head   = %d\n", __func__, hparams.n_text_head);
        fprintf(stderr, "%s: n_text_layer  = %d\n", __func__, hparams.n_text_layer);
        fprintf(stderr, "%s: n_mels        = %d\n", __func__, hparams.n_mels);
        fprintf(stderr, "%s: ftype (src)   = %d\n", __func__, hparams.ftype);
        fprintf(stderr, "%s: qntvr (src)   = %d\n", __func__, qntvr_src);
        fprintf(stderr, "%s: ftype (dst)   = %d\n", __func__, ftype_dst);
        fprintf(stderr, "%s: qntvr (dst)   = %d\n", __func__, GGML_QNT_VERSION);

        fout.write((const char *) &hparams.n_vocab,       sizeof(hparams.n_vocab));
        fout.write((const char *) &hparams.n_audio_ctx,   sizeof(hparams.n_audio_ctx));
        fout.write((const char *) &hparams.n_audio_state, sizeof(hparams.n_audio_state));
        fout.write((const char *) &hparams.n_audio_head,  sizeof(hparams.n_audio_head));
        fout.write((const char *) &hparams.n_audio_layer, sizeof(hparams.n_audio_layer));
        fout.write((const char *) &hparams.n_text_ctx,    sizeof(hparams.n_text_ctx));
        fout.write((const char *) &hparams.n_text_state,  sizeof(hparams.n_text_state));
        fout.write((const char *) &hparams.n_text_head,   sizeof(hparams.n_text_head));
        fout.write((const char *) &hparams.n_text_layer,  sizeof(hparams.n_text_layer));
        fout.write((const char *) &hparams.n_mels,        sizeof(hparams.n_mels));
        fout.write((const char *) &ftype_dst,             sizeof(hparams.ftype));
    }

    // Load mel filters
    {
        whisper_filters filters;

        finp.read ((char *) &filters.n_mel, sizeof(filters.n_mel));
        fout.write((char *) &filters.n_mel, sizeof(filters.n_mel));
        finp.read ((char *) &filters.n_fft, sizeof(filters.n_fft));
        fout.write((char *) &filters.n_fft, sizeof(filters.n_fft));

        filters.data.resize(filters.n_mel * filters.n_fft);
        finp.read ((char *) filters.data.data(), filters.data.size() * sizeof(float));
        fout.write((char *) filters.data.data(), filters.data.size() * sizeof(float));
    }

    // Load vocab - just copy bytes through
    {
        int32_t n_vocab = 0;
        finp.read ((char *) &n_vocab, sizeof(n_vocab));
        fout.write((char *) &n_vocab, sizeof(n_vocab));

        char word[129];

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            finp.read ((char *) &len, sizeof(len));
            fout.write((char *) &len, sizeof(len));

            finp.read ((char *) word, len);
            fout.write((char *) word, len);
        }
    }

    // Regexes of tensor names to not be quantized
    const std::vector<std::string> to_quant = { ".*" };
    const std::vector<std::string> to_skip = {
        //"encoder.*",
        "encoder.conv1.bias",
        "encoder.conv2.bias",
        "encoder.positional_embedding",
        "decoder.positional_embedding",
    };

    // --- First pass: count tensors ---
    const auto tensors_start_pos = finp.tellg();
    int total_tensors = 0;
    {
        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            finp.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            finp.read(reinterpret_cast<char *>(&length), sizeof(length));
            finp.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));

            if (finp.eof()) {
                break;
            }

            // Skip dimension values
            int32_t nelements = 1;
            int32_t ne[4] = { 1, 1, 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                finp.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            // Skip tensor name
            finp.seekg(length, std::ios::cur);

            // Skip tensor data
            const int bpe = (ttype == 0) ? sizeof(float) : sizeof(uint16_t);
            finp.seekg((std::streamoff)nelements * bpe, std::ios::cur);

            total_tensors++;
        }

        // Seek back to start of tensors
        finp.clear(); // clear EOF flag
        finp.seekg(tensors_start_pos);
    }

    fprintf(stderr, "%s: found %d tensors to process\n", __func__, total_tensors);

    if (progress_callback) {
        progress_callback(0.0f);
    }

    // --- Second pass: quantize with per-tensor progress ---

    // Resolve target quantization type
    ggml_type qtype = GGML_TYPE_F32;

    switch (ftype) {
        case GGML_FTYPE_MOSTLY_Q4_0: qtype = GGML_TYPE_Q4_0; break;
        case GGML_FTYPE_MOSTLY_Q4_1: qtype = GGML_TYPE_Q4_1; break;
        case GGML_FTYPE_MOSTLY_Q5_0: qtype = GGML_TYPE_Q5_0; break;
        case GGML_FTYPE_MOSTLY_Q5_1: qtype = GGML_TYPE_Q5_1; break;
        case GGML_FTYPE_MOSTLY_Q8_0: qtype = GGML_TYPE_Q8_0; break;
        case GGML_FTYPE_MOSTLY_Q2_K: qtype = GGML_TYPE_Q2_K; break;
        case GGML_FTYPE_MOSTLY_Q3_K: qtype = GGML_TYPE_Q3_K; break;
        case GGML_FTYPE_MOSTLY_Q4_K: qtype = GGML_TYPE_Q4_K; break;
        case GGML_FTYPE_MOSTLY_Q5_K: qtype = GGML_TYPE_Q5_K; break;
        case GGML_FTYPE_MOSTLY_Q6_K: qtype = GGML_TYPE_Q6_K; break;
        default: {
            fprintf(stderr, "%s: invalid model type %d\n", __func__, ftype);
            return WHISPER_QUANTIZE_ERROR_QUANTIZATION_FAILED;
        }
    }

    if (!ggml_is_quantized(qtype)) {
        fprintf(stderr, "%s: invalid quantization type %d (%s)\n", __func__, qtype, ggml_type_name(qtype));
        return WHISPER_QUANTIZE_ERROR_QUANTIZATION_FAILED;
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<float> work;

    std::vector<uint8_t>     data_u8;
    std::vector<ggml_fp16_t> data_f16;
    std::vector<float>       data_f32;

    int tensor_idx = 0;

    while (true) {
        int32_t n_dims;
        int32_t length;
        int32_t ttype;

        finp.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        finp.read(reinterpret_cast<char *>(&length), sizeof(length));
        finp.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));

        if (finp.eof()) {
            break;
        }

        int32_t nelements = 1;
        int32_t ne[4] = { 1, 1, 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            finp.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            nelements *= ne[i];
        }

        std::string name(length, 0);
        finp.read(&name[0], length);

        printf("%64s - [%5d, %5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ne[2], ggml_type_name((ggml_type) ttype));

        bool quantize = false;

        // Check if we should quantize this tensor
        for (const auto & s : to_quant) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = true;
                break;
            }
        }

        // Check if we should skip this tensor
        for (const auto & s : to_skip) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = false;
                break;
            }
        }

        // Quantize only 2D tensors
        quantize &= (n_dims == 2);

        if (quantize) {
            if (ttype != GGML_TYPE_F32 && ttype != GGML_TYPE_F16) {
                fprintf(stderr, "%s: unsupported ttype %d (%s) for integer quantization\n", __func__, ttype, ggml_type_name((ggml_type) ttype));
                return WHISPER_QUANTIZE_ERROR_QUANTIZATION_FAILED;
            }

            if (ttype == GGML_TYPE_F16) {
                data_f16.resize(nelements);
                finp.read(reinterpret_cast<char *>(data_f16.data()), nelements * sizeof(ggml_fp16_t));
                data_f32.resize(nelements);
                for (int i = 0; i < nelements; ++i) {
                    data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                }
            } else {
                data_f32.resize(nelements);
                finp.read(reinterpret_cast<char *>(data_f32.data()), nelements * sizeof(float));
            }

            ttype = qtype;
        } else {
            const int bpe = (ttype == 0) ? sizeof(float) : sizeof(uint16_t);

            data_u8.resize(nelements*bpe);
            finp.read(reinterpret_cast<char *>(data_u8.data()), nelements * bpe);
        }

        fout.write(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        fout.write(reinterpret_cast<char *>(&length), sizeof(length));
        fout.write(reinterpret_cast<char *>(&ttype),  sizeof(ttype));
        for (int i = 0; i < n_dims; ++i) {
            fout.write(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        }
        fout.write(&name[0], length);

        if (quantize) {
            work.resize(nelements); // for quantization

            size_t cur_size = 0;
            switch ((ggml_type) ttype) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    {
                        cur_size = ggml_quantize_chunk((ggml_type) ttype, data_f32.data(), work.data(), 0, nelements/ne[0], ne[0], nullptr);
                    } break;
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_I8:
                case GGML_TYPE_I16:
                case GGML_TYPE_I32:
                case GGML_TYPE_I64:
                case GGML_TYPE_F64:
                case GGML_TYPE_Q8_1:
                case GGML_TYPE_Q8_K:
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                case GGML_TYPE_IQ3_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ1_S:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                case GGML_TYPE_IQ1_M:
                case GGML_TYPE_BF16:
                case GGML_TYPE_TQ1_0:
                case GGML_TYPE_TQ2_0:
                case GGML_TYPE_MXFP4:
                case GGML_TYPE_COUNT:
                    {
                        fprintf(stderr, "%s: unsupported quantization type %d (%s)\n", __func__, ttype, ggml_type_name((ggml_type) ttype));
                        return WHISPER_QUANTIZE_ERROR_QUANTIZATION_FAILED;
                    }
            }

            fout.write(reinterpret_cast<char *>(work.data()), cur_size);
            total_size_new += cur_size;

            printf("size = %8.2f MB -> %8.2f MB\n", nelements * sizeof(float)/1024.0/1024.0, cur_size/1024.0/1024.0);
        } else {
            printf("size = %8.3f MB\n", data_u8.size()/1024.0/1024.0);
            fout.write(reinterpret_cast<char *>(data_u8.data()), data_u8.size());
            total_size_new += data_u8.size();
        }

        total_size_org += nelements * sizeof(float);
        tensor_idx++;

        // Report per-tensor progress
        if (progress_callback && total_tensors > 0) {
            progress_callback((float)tensor_idx / (float)total_tensors);
        }
    }

    printf("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    printf("%s: quant size  = %8.2f MB | ftype = %d (%s)\n", __func__, total_size_new/1024.0/1024.0, ftype, ggml_type_name(qtype));

    finp.close();
    fout.close();

    return WHISPER_QUANTIZE_OK;
}

// Main quantization function exposed to Rust
int whisper_model_quantize(
    const char * fname_inp,
    const char * fname_out,
    int ftype,
    whisper_quantize_progress_callback progress_callback) {

    // Validate ftype
    ggml_ftype ggml_ftype_value = (ggml_ftype)ftype;

    // Check if it's a valid quantization type
    switch (ggml_ftype_value) {
        case GGML_FTYPE_MOSTLY_Q4_0:
        case GGML_FTYPE_MOSTLY_Q4_1:
        case GGML_FTYPE_MOSTLY_Q5_0:
        case GGML_FTYPE_MOSTLY_Q5_1:
        case GGML_FTYPE_MOSTLY_Q8_0:
        case GGML_FTYPE_MOSTLY_Q2_K:
        case GGML_FTYPE_MOSTLY_Q3_K:
        case GGML_FTYPE_MOSTLY_Q4_K:
        case GGML_FTYPE_MOSTLY_Q5_K:
        case GGML_FTYPE_MOSTLY_Q6_K:
            break;
        default:
            fprintf(stderr, "%s: invalid quantization type %d\n", __func__, ftype);
            return WHISPER_QUANTIZE_ERROR_INVALID_FTYPE;
    }

    // Initialize GGML (needed for f16 tables)
    static bool ggml_initialized = false;
    if (!ggml_initialized) {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
        ggml_initialized = true;
    }

    return whisper_model_quantize_internal(
        fname_inp,
        fname_out,
        ggml_ftype_value,
        progress_callback
    );
}

// Get the quantization type of a model file
int whisper_model_get_ftype(const char * fname) {
    std::ifstream fin(fname, std::ios::binary);
    if (!fin) {
        return -1;
    }

    // Check magic
    uint32_t magic;
    fin.read((char *) &magic, sizeof(magic));
    if (magic != GGML_FILE_MAGIC) {
        return -1;
    }

    // Skip to ftype field
    whisper_model_hparams hparams = {};
    fin.read((char *) &hparams.n_vocab,       sizeof(hparams.n_vocab));
    fin.read((char *) &hparams.n_audio_ctx,   sizeof(hparams.n_audio_ctx));
    fin.read((char *) &hparams.n_audio_state, sizeof(hparams.n_audio_state));
    fin.read((char *) &hparams.n_audio_head,  sizeof(hparams.n_audio_head));
    fin.read((char *) &hparams.n_audio_layer, sizeof(hparams.n_audio_layer));
    fin.read((char *) &hparams.n_text_ctx,    sizeof(hparams.n_text_ctx));
    fin.read((char *) &hparams.n_text_state,  sizeof(hparams.n_text_state));
    fin.read((char *) &hparams.n_text_head,   sizeof(hparams.n_text_head));
    fin.read((char *) &hparams.n_text_layer,  sizeof(hparams.n_text_layer));
    fin.read((char *) &hparams.n_mels,        sizeof(hparams.n_mels));
    fin.read((char *) &hparams.ftype,         sizeof(hparams.ftype));

    fin.close();

    // Extract the actual ftype (without version)
    return hparams.ftype % GGML_QNT_VERSION_FACTOR;
}

} // extern "C"