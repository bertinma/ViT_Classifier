from d2l import torch as d2l
from model import transformer
import argparse
import torch 

def train(opt):
    model = transformer.ViT(opt.img_size, opt.patch_size, opt.num_hiddens, opt.mlp_num_hiddens, opt.num_heads,
                opt.num_blocks, opt.emb_dropout, opt.block_dropout, opt.lr)
    data = d2l.FashionMNIST(batch_size=8, resize=(opt.img_size, opt.img_size))
    trainer = d2l.Trainer(max_epochs=10, num_gpus=0)
    trainer.fit(model, data)
    return model

def save_model(model, path="weights/vit.pt"):
    torch.save(model.state_dict(), path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=96, help="Size of image")
    parser.add_argument('--patch-size', type=int, default=16, help="Size of patches")
    parser.add_argument('--num-hiddens', type=int, default=512)
    parser.add_argument('--mlp-num-hiddens', type=int, default=2048)
    parser.add_argument('--num-heads', type=int, default=8, help="Number of head attentions")
    parser.add_argument('--num-blocks', type=int, default=2, help="Number of blocks")
    parser.add_argument('--emb-dropout', type=float, default=.1, help="Embedded dropout")
    parser.add_argument('--block-dropout', type=float, default=.1, help="Block dropout")
    parser.add_argument('--lr', type=float, default=.1, help="Learning rate")
    opt = parser.parse_args()


    opt.img_size, opt.patch_size = 96, 16
    opt.num_hiddens, opt.mlp_num_hiddens, opt.num_heads, opt.num_blocks = 512, 2048, 8, 2
    opt.emb_dropout, opt.block_dropout, opt.lr = 0.1, 0.1, 0.1
    
    model = train(opt)
    save_model(model)