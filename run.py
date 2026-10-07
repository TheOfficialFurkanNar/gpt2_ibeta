import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from config import ChatConfig


class CustomStopCriteria(StoppingCriteria):
    def __init__(self, stop_ids):
        super().__init__()
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_id = input_ids[0][-1].item()
        return last_id in self.stop_ids


class CLIChat:
    def __init__(self, config: ChatConfig):
        # Check for CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This application requires a CUDA-enabled GPU.")
        
        self.config = config
        self.device = torch.device("cuda")
        print(f"Using device: {self.device}")
        
        # Use local path if provided, otherwise use model name
        model_path = config.local_model_path if config.local_model_path else config.model_name
        print(f"Loading model from: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        self.conversation_history = []
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Compute stop token IDs if any stop sequence can be mapped to single tokens
        stop_ids = [self.tokenizer.eos_token_id]
        for seq in self.config.stop_sequences:
            encoded = self.tokenizer.encode(seq, add_special_tokens=False)
            if len(encoded) == 1:
                stop_ids.append(encoded[0])
        self.stopping_criteria = StoppingCriteriaList([CustomStopCriteria(stop_ids)])
        
        print("Model loaded successfully!\n")
    
    def build_prompt(self, user_input: str) -> str:
        """Construct multi-turn instruction prompt adhering to Dolly-15k/Alpaca formats."""
        prompt_parts = []
        if self.config.system_prompt:
            prompt_parts.append(self.config.system_prompt)
        
        # Append historical turns
        for user_msg, ai_msg in self.conversation_history:
            prompt_parts.append(f"\n\nInstruction:\n{user_msg}\n\nResponse:\n{ai_msg}")
        
        # Append current user prompt
        prompt_parts.append(f"\n\nInstruction:\n{user_input}\n\nResponse:\n")
        
        full_prompt = "".join(prompt_parts).strip()

        # Tokenize and trim context if prompt exceeds max_length
        tokens = self.tokenizer.encode(full_prompt)
        max_allowed = self.config.max_length - self.config.max_new_tokens
        if len(tokens) > max_allowed and len(self.conversation_history) > 0:
            # If history is too long, drop earlier turns
            self.conversation_history.pop(0)
            return self.build_prompt(user_input)
            
        return full_prompt

    def truncate_at_stop_sequences(self, text: str) -> str:
        """Truncate generated output at the first occurrence of any stop sequence."""
        earliest_idx = len(text)
        for stop_seq in self.config.stop_sequences:
            idx = text.find(stop_seq)
            if idx != -1 and idx < earliest_idx:
                earliest_idx = idx
        return text[:earliest_idx].strip()

    def generate_response(self, user_input: str) -> str:
        full_prompt = self.build_prompt(user_input)
        
        inputs = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        input_length = inputs.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=self.stopping_criteria
            )
        
        # Extract generated tokens
        generated_tokens = outputs[0][input_length:]
        
        # Truncate at EOS token if present
        eos_positions = (generated_tokens == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            generated_tokens = generated_tokens[:eos_positions[0].item()]
        
        raw_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Post-process truncation for stop sequences
        clean_response = self.truncate_at_stop_sequences(raw_response)
        
        return clean_response
    
    def chat(self):
        print("=== CLI Chat with GPT-2 Instruct Model ===")
        print("Type 'quit' or 'exit' to end the conversation\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            # Generate response
            print("AI: ", end="", flush=True)
            response = self.generate_response(user_input)
            print(response)
            
            # Record turn in history as tuple (User, AI)
            self.conversation_history.append((user_input, response))
            print()


def main():
    config = ChatConfig()
    chat = CLIChat(config)
    chat.chat()


if __name__ == "__main__":
    main()
