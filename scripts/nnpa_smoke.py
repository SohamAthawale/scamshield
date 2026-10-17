import time, torch, torch_nnpa
print("available", torch_nnpa.NNPAModule.is_available(), "telum2", torch_nnpa.NNPAModule.is_telum2_available())
dev = torch_nnpa.device(); print("device", dev, type(dev))
torch.manual_seed(0)
m = torch.nn.Sequential(torch.nn.Linear(19,12), torch.nn.ReLU(), torch.nn.Linear(12,6), torch.nn.ReLU(),
                        torch.nn.Linear(6,12), torch.nn.ReLU(), torch.nn.Linear(12,19)).eval()
def bench(model, x, n=2000):
    with torch.no_grad():
        for _ in range(50): y = model(x)
        if hasattr(y, "cpu"): y.cpu()
        ts = []
        for _ in range(n):
            t = time.perf_counter(); y = model(x); y = y.cpu(); ts.append(time.perf_counter() - t)
    ts.sort(); return y, ts[len(ts)//2]*1e6, ts[int(len(ts)*.99)]*1e6
for bs in (1, 32, 256, 4096):
    x = torch.randn(bs, 19)
    yc, p50c, p99c = bench(m, x)
    try:
        import copy; mn = copy.deepcopy(m).to(dev)
        yn, p50n, p99n = bench(mn, x.to(dev))
        print(f"bs={bs} cpu p50={p50c:.1f}us p99={p99c:.1f}us | nnpa p50={p50n:.1f}us p99={p99n:.1f}us | maxdiff={(yc-yn).abs().max().item():.2e}")
    except Exception as e:
        print("bs", bs, "nnpa failed:", repr(e)[:400])
try:
    print("to('nnpa'):", torch.randn(2,2).to("nnpa").device)
except Exception as e: print("to('nnpa') failed", repr(e)[:200])
