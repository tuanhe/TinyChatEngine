#include <iostream>
#include <map>
#include <string>
#include <cstring>
#include <unistd.h>
#include <getopt.h>
#include "Generate.h"
#include "interface.h"

std::map<std::string, int> model_config = {
    {"OPT_125m", OPT_125M},         {"OPT_1.3B", OPT_1_3B},               {"OPT_6.7B", OPT_6_7B}, {"LLaMA_7B", LLaMA_7B},
    {"7b", LLaMA_7B},               {"LLaMA2_7B_chat", LLaMA_7B},         {"13b", LLaMA_13B},     {"LLaMA2_13B_chat", LLaMA_13B},
    {"CodeLLaMA_7B_Instruct", CodeLLaMA_7B},                              {"CodeLLaMA_13B_Instruct", CodeLLaMA_13B}, 
    {"StarCoder", StarCoder_15_5B}, {"StarCoder_15.5B", StarCoder_15_5B}, {"LLaVA_7B", LLaVA_7B}, {"LLaVA_13B", LLaVA_13B}, 
    {"VILA_2.7B", VILA_2_7B},       {"VILA_7B", VILA_7B},                 {"VILA_13B", VILA_13B}, {"Clip_ViT_Large", Clip_ViT_Large}, 
    {"Mistral_7B", Mistral_7B}
    };

std::map<std::string, std::string> default_model_path = {{"OPT_125m", "models/OPT_125m"},
                                                 {"OPT_1.3B", "models/OPT_1.3B"},
                                                 {"OPT_6.7B", "models/OPT_6.7B"},
                                                 {"LLaMA_7B", "models/LLaMA_7B"},
                                                 {"LLaMA2_7B_chat", "models/LLaMA_7B_2_chat"},
                                                 {"LLaMA2_13B_chat", "models/LLaMA_13B_2_chat"},
                                                 {"7b", "models/LLaMA_7B_2_chat"},
                                                 {"13b", "models/LLaMA_13B_2_chat"},
                                                 {"CodeLLaMA_7B_Instruct", "models/CodeLLaMA_7B_Instruct"},
                                                 {"CodeLLaMA_13B_Instruct", "models/CodeLLaMA_13B_Instruct"},
                                                 {"StarCoder", "models/StarCoder"},
                                                 {"StarCoder_15.5B", "models/StarCoder"},
                                                 {"LLaVA_7B", "models/LLaVA_7B"},
                                                 {"LLaVA_13B", "models/LLaVA_13B"},
                                                 {"VILA_2.7B", "models/VILA_2.7B"},
                                                 {"VILA_7B", "models/VILA_7B"},
                                                 {"VILA_13B", "models/VILA_13B"},
                                                 {"Clip_ViT_Large", "models/CLIP_ViT_Large"},
                                                 {"Mistral_7B", "models/Mistral_7B"},
                                                 };

std::map<std::string, int> data_format_list = {
    {"FP32", FP32}, {"INT8", QINT8}, {"INT4", INT4}, {"int4", INT4}, {"fp32", FP32},
};

bool isLLaMA(std::string s) {
    std::string LLaMA_prefix = "LLaMA";
    std::string CodeLLaMA_prefix = "CodeLLaMA";
    if (s.substr(0, LLaMA_prefix.size()) == LLaMA_prefix || s.substr(0, CodeLLaMA_prefix.size()) == CodeLLaMA_prefix || s == "7b" || s == "13b")
        return true;
    else
        return false;
}

bool isCodeLLaMA(std::string s) {
    std::string CodeLLaMA_prefix = "CodeLLaMA";
    if (s.substr(0, CodeLLaMA_prefix.size()) == CodeLLaMA_prefix)
        return true;
    else
        return false;
}

bool isStarCoder(std::string s) {
    std::string StarCoder_prefix = "StarCoder";
    if (s.substr(0, StarCoder_prefix.size()) == StarCoder_prefix)
        return true;
    else
        return false;
}

bool isLLaVA(std::string s) {
    std::string LLaVA_prefix = "LLaVA";
    if (s.substr(0, LLaVA_prefix.size()) == LLaVA_prefix)
        return true;
    else
        return false;
}

bool isVILA(std::string s) {
    std::string VILA_prefix = "VILA";
    if (s.substr(0, VILA_prefix.size()) == VILA_prefix)
        return true;
    else
        return false;
}

bool isMistral(std::string s) {
    std::string Mistral_prefix = "Mistral";
    if (s.substr(0, Mistral_prefix.size()) == Mistral_prefix)
        return true;
    else
        return false;
}

bool convertToBool(const char* str) {
    if (strcmp(str, "true") == 0 || strcmp(str, "1") == 0) {
        return true;
    }
    else if (strcmp(str, "false") == 0 || strcmp(str, "0") == 0) {
        return false;
    }
    else {
        std::cerr << "Error: Invalid boolean value: " << str << std::endl;
        exit(EXIT_FAILURE);
    }
}

void arguments_check(std::string& model_name, std::string model_data_type)
{
    if (model_config.count(model_name) == 0) {
        std::cerr << "Model config:" << model_name << " unsupported" << std::endl;
        std::cerr << "Please select one of the following:";
        for (const auto& k : model_config) {
            std::cerr << k.first << ", ";
        }
        std::cerr << std::endl;
        throw("Unsupported model\n");
    }
    std::cout << "Using model: " << model_name << std::endl;

    auto data_format_input = model_data_type;
    if (data_format_list.count(data_format_input) == 0) {
        std::cerr << "Data format:" << data_format_input << " unsupported" << std::endl;
        std::cerr << "Please select one of the following: ";
        for (const auto& k : data_format_list) {
            std::cerr << k.first << ", ";
        }
        std::cerr << std::endl;
        throw("Unsupported data format\n");
    }
    
    if (model_data_type == "INT4" || model_data_type == "int4")
        std::cout << "Using AWQ for 4bit quantization: https://github.com/mit-han-lab/llm-awq" << std::endl;
    else
        std::cout << "Using data format: " << model_data_type << std::endl;
    return ;
}

int main(int argc, char* argv[]) {
    int opt;
    std::string arg;
    std::string model_name = "LLaMA2_7B_chat";
    std::string model_path;
    std::string model_data_type = "INT4";
    int num_thread = 6;
    bool use_voicechat = false;

    while ((opt = getopt(argc, argv, "m:p:t:j")) != -1) {
        switch (opt) {
            case 'm':
                model_name = optarg;
                break;
            case 'p':
                model_path = optarg;
                break;
            case 't':
                model_data_type = optarg;
                break;
            case 'j':
                num_thread = std::stoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -m <model path>" << std::endl;
                exit(EXIT_FAILURE);
        }
    }

    
    arguments_check(model_name, model_data_type);
    
    bool instruct = true;
    std::string img_path = "images/monalisa.jpg";
    Profiler::getInstance().for_demo = true;

    // Set prompt color
    set_print_yellow();
    std::cout << "TinyChatEngine by MIT HAN Lab: https://github.com/mit-han-lab/TinyChatEngine" << std::endl;
    

    if (isLLaMA(model_name)) {
        int format_id = data_format_list[model_data_type];

        // Voicechat instructions
        if (use_voicechat) {
            std::cout << "You are using the TinyVoiceChat." << std::endl;
            std::cout << "*Usage instructions*" << std::endl;
            std::cout << "- Please use this mode in a quiet environment to have a better user experience and avoid speech misdetection." << std::endl;
            std::cout << "- Please start speaking after \"USER: [Start speaking]\" shows up." << std::endl;
            std::cout << "- Please press `Ctrl+C` multiple times to exit the program." << std::endl << std::endl;
        }

        // Load model
        std::cout << "Loading model... " << std::flush;
        int model_id = model_config[model_name];
        std::string m_path = default_model_path[model_name];

        std::cout << "m_path " << m_path << std::endl;

        struct opt_params generation_config;
        generation_config.n_predict = 512;
        generation_config.repeat_penalty = 1.1f;
        generation_config.temp = 0.2f;
        if(isCodeLLaMA(model_name)) {
            generation_config.n_vocab = 32016;
        }
        else {
            generation_config.n_vocab = 32000;
        }

        bool first_prompt = true;

        if (format_id == FP32) {
            Fp32LlamaForCausalLM model = Fp32LlamaForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl << std::endl;

            // Get input from the user
            while (true) {
                std::string input;
                if (use_voicechat) {
                    // Set prompt color
                    set_print_yellow();
                    int result = std::system("./application/sts_utils/listen");
                    std::ifstream in("tmpfile");
                    // set user input color
                    set_print_red();
                    std::getline(in, input);
                    result = std::system("rm tmpfile");
                    (void)result;
                    std::cout << input << std::endl;
                    // reset color
                    set_print_reset();
                } else {
                    // Set prompt color
                    set_print_yellow();
                    std::cout << "USER: ";
                    // set user input color
                    set_print_red();
                    std::getline(std::cin, input);
                    // reset color
                    set_print_reset();
                }
                if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.")
                    break;
                if (instruct) {
                    std::cout << "ASSISTANT: ";
                    if (isCodeLLaMA(model_name)) {
                        if (first_prompt) {
                            input = "<s>[INST] " + input + " [/INST] ";
                            first_prompt = false;
                        }
                        else {
                            input = " </s> <s>[INST] " + input + " [/INST] ";
                        }
                    }
                }
                else {
                    if (isCodeLLaMA(model_name)) {
                        std::cout << input;
                    }
                }

                if (!isCodeLLaMA(model_name)) {
                    if (first_prompt) {
                        input = "A chat between a human and an assistant.\n\n### Human: " + input + "\n### Assistant: \n";
                        first_prompt = false;
                    }
                    else {
                        input = "### Human: " + input + "\n### Assistant: \n";
                    }
                }

                LLaMAGenerate(m_path, &model, LLaMA_FP32, input, generation_config, "models/llama_vocab.bin", true, false);
            }
        } else if (format_id == INT4) {
            m_path = model_path + m_path;
            std::string voc_path = model_path + "../models/llama_vocab.bin";
            Int4LlamaForCausalLM model = Int4LlamaForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl << std::endl;
            
            // Get input from the user
            while (true) {
                std::string input;
                if (use_voicechat) {
                    // Set prompt color
                    set_print_yellow();
                    int result = std::system("./application/sts_utils/listen");
                    std::ifstream in("tmpfile");
                    // set user input color
                    set_print_red();
                    std::getline(in, input);
                    result = std::system("rm tmpfile");
                    (void)result;
                    std::cout << input << std::endl;
                    // reset color
                    set_print_reset();
                } else {
                    // Set prompt color
                    set_print_yellow();
                    std::cout << "USER: ";
                    // set user input color
                    set_print_red();
                    std::getline(std::cin, input);
                    // reset color
                    set_print_reset();
                }
                if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.")
                    break;
                if (instruct) {
                    std::cout << "ASSISTANT: ";
                    if (isCodeLLaMA(model_name)) {
                        if (first_prompt) {
                            input = "<s>[INST] " + input + " [/INST] ";
                            first_prompt = false;
                        }
                        else {
                            input = " </s> <s>[INST] " + input + " [/INST] ";
                        }
                    }
                }
                else {
                    if (isCodeLLaMA(model_name)) {
                        std::cout << input;
                    }
                }

                if (!isCodeLLaMA(model_name)) {
                    if (first_prompt) {
                        input = "A chat between a human and an assistant.\n\n### Human: " + input + "\n### Assistant: \n";
                        first_prompt = false;
                    }
                    else {
                        input = "### Human: " + input + "\n### Assistant: \n";
                    }
                }
                LLaMAGenerate(m_path, &model, LLaMA_INT4, input, generation_config, voc_path, true, use_voicechat);
            }
        } else {
            std::cout << std::endl;
            std::cerr << "At this time, we only support FP32 and INT4 for LLaMA_7B." << std::endl;
        }
    } 
};
