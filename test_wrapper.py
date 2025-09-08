from model.model_wrapper import *
import argparse
import torch
import torch.nn as nn
from transformers import AutoProcessor

def run_test(device: str):
    """
    Runs the full test suite for the Smollm_custom distillation implementation.
    """
    print(f"--- Running test on device: {device} ---")

    # =========================================================================
    # 1. Configuration
    # =========================================================================
    model_name = "/home/jinkaiyan/MaTVLM/smolVLM/SmolVLM-Intruct"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Configuration for the Mamba layers to be inserted
    target_layers = [0]
    mamba_cfg = PhiMambaConfig(
        2048,
        {"expand": 1, "ngroups": 32, "d_state": 64, "d_conv": 4},
        1e-05,
        d_inner=2048,
        d_xb=2048,
        intermediate_size=8192,
        hidden_act="silu",
        n_layer=24,
        attn_layers=[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23],
        resid_pdrop=0.1,
        bidirectional=False,
        is_bias=False
    )

    # =========================================================================
    # 2. Model and Processor Initialization
    # =========================================================================
    print(f"\n--- 1. Initializing models and processor from '{model_name}' ---")
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
    print(processor)
    # processor.to(device)

    # Initialize Student Model (with Mamba layers)
    attn_layers = [0 , 1, 2, 3 ,5,6,7,8,9,10,11,13,14,15,16,17,18,19,21,22,23]
    teacher_model = HybridSmolVLMForConditionalGeneration.from_pretrained(model_name,torch_dtype=dtype)
    student_model = HybridSmolVLMWrapper.init_distillation(
        checkpoint_path=None,
        tranformer_name=model_name,
        mamba_config=mamba_cfg,
        attn_layers=attn_layers,
        dtype=dtype,
    )
    # student_model.gradie
    # print(student_model.model)
    student_model.model.from_pretrained(model_name, torch_dtype=dtype, attn_implementation="flash_attention_2")
    student_model.train()  # Set to train mode for loss calculation
    student_model.to(device)
    print("Student model initialized successfully.")

    # Initialize Teacher Model (standard, unmodified)

    teacher_model.to(device)
    teacher_model.eval()  # Teacher is only for inference
    print("Teacher model initialized successfully.")

    # =========================================================================
    # 3. Prepare Dummy Multimodal Input
    # =========================================================================
    print("\n--- 2. Preparing dummy multimodal input ---")
    from transformers.image_utils import load_image
    dummy_image = load_image('/home/jinkaiyan/MaTVLM/smolVLM/images/bee.jpg')
    dummy_prompt = "<image>\nWhat is in the photo?."

    inputs = processor(text=[dummy_prompt],
                       images=[[dummy_image]], return_tensors="pt",padding=True).to(device)
    inputs.to(dtype)
    print("Input shapes:", {k: v.shape for k, v in inputs.items()})
    print()

    # =========================================================================
    # 4. Test Distillation Forward Pass
    # =========================================================================
    print("\n--- 3. Testing distillation forward pass ---")

    # Step 4a: Get teacher's hidden states
    print("  - Running teacher forward pass...")
    with torch.no_grad():
        teacher_full_outputs = teacher_model(**inputs, output_hidden_states=True)
        teacher_hidden_states = teacher_full_outputs.hidden_states

    print(f"  - Teacher produced {len(teacher_hidden_states)} hidden states. and the shape of each hidden state is {teacher_hidden_states[1].shape}")
    assert len(teacher_hidden_states) == mamba_cfg.n_layer + 1, "Teacher should output n_layer+1 hidden states"

    # Step 4b: Run student's custom forward pass
    print("  - Running student forward pass with teacher outputs...")
    student_distill_output = student_model(**inputs,teacher_outputs=teacher_hidden_states)
    # print(student_distill_output)

    # Step 4c: Verify the output structure and shapes
    # print("  - Verifying output structure and shapes...")
    #
    # assert isinstance(student_distill_output.hidden_states, tuple) and len(student_distill_output.hidden_states) == 2, \
    #     "Output hidden_states should be a tuple of (student_states, teacher_states)"
    #
    student_states= student_distill_output.hidden_states
    #
    assert len(student_states) == mamba_cfg.n_layer + 1, "Student should have n_layer+1 hidden states"
    # assert len(processed_teacher_states) == mamba_cfg.n_layer, "Processed teacher should have n_layer hidden states"
    #
    # # Check that shapes match
    # # We compare the student's state *after* a layer with the teacher's state *after* being processed by that layer
    # assert student_states[1].shape == processed_teacher_states[0].shape, "Hidden state shapes do not match"
    # print("  - Output structure and shapes are correct.")

    # Step 4d: Verify loss calculations
    print("  - Verifying loss calculations...")
    lm_loss = 1


    loss_fct = nn.MSELoss()
    distill_loss = 0.0
    for s_state, t_state in zip(student_states[1:], teacher_hidden_states[1:]):
        distill_loss += loss_fct(s_state, t_state)

    assert distill_loss.requires_grad, "Distillation loss must have grad"
    print(f"  - Distillation Loss (MSE): {distill_loss.item():.4f}")

    # Check that backpropagation would work
    total_loss = 0.5 * lm_loss + 0.5 * distill_loss
    total_loss.backward()
    print("  - Backward pass on total loss successful.")
    student_model.zero_grad()

    # =========================================================================
    # 5. Test Standard Inference (Generation)
    # =========================================================================
    print("\n--- 4. Testing standard inference (generate) ---")
    student_model.eval()  # Switch to eval mode for generation
    with torch.inference_mode():
        gen_ids = student_model.generate(
            **inputs,
            max_new_tokens=50,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    decoded_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    print(f"  - Generated text: \"{decoded_text.strip()}\"")
    assert len(decoded_text) > len(dummy_prompt), "Generation should produce new text"
    print("  - Generation successful.")

    print("\n--- All tests passed successfully! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for Smollm_custom distillation.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run the test on."
    )
    args = parser.parse_args()
    run_test(args.device)




