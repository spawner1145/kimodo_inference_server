"""Real tiny NF4 serialization/loading and block swap, no downloaded weights."""
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from kimodo_server.encoder import NF4Encoder


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class NF4Tests(unittest.TestCase):
    def test_prequantized_llm2vec_and_block_swap(self):
        try:
            import bitsandbytes
        except ImportError:
            self.skipTest('Optional bitsandbytes is not installed')
        from transformers import LlamaConfig,LlamaForCausalLM,BitsAndBytesConfig
        # Reuse local tokenizer vocabulary only. The model is random and tiny,
        # and makes no claim about motion generation quality.
        tokenizer = Path(__file__).resolve().parents[1]/'models/text_encoders/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp'
        if not (tokenizer/'tokenizer.json').exists():
            self.skipTest('Local Llama tokenizer is not available')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base,packed = root/'base',root/'packed'
            config = LlamaConfig(vocab_size=128256,hidden_size=64,intermediate_size=128,
                num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                max_position_embeddings=512,use_cache=False)
            torch.manual_seed(31)
            LlamaForCausalLM(config).save_pretrained(base)
            quant = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
            model = LlamaForCausalLM.from_pretrained(base,quantization_config=quant,device_map='cuda',dtype=torch.bfloat16)
            model.save_pretrained(packed)
            del model
            for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json'):
                if (tokenizer/name).exists():
                    shutil.copy2(tokenizer/name,packed/name)
            encoder = NF4Encoder(packed)
            encoder.to('cuda')
            reference = encoder.encode('A person walks forward.')
            self.assertEqual(reference.shape,(1,64))
            self.assertTrue(torch.isfinite(reference).all())
            encoder.to('cpu')
            encoder.swapper.close()
            encoder.setup_residency(2)
            encoder.to('cuda')
            actual = encoder.encode('A person walks forward.')
            torch.testing.assert_close(actual,reference,atol=0,rtol=0)
            self.assertEqual(encoder.swapper.transfers,2)
            self.assertTrue(all(p.device.type=='cpu' for block in encoder.swapper.blocks for p in block.parameters()))
            encoder.to('cpu')
            encoder.close()


if __name__=='__main__':
    unittest.main()
