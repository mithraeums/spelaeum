# Spelaeum — many machines training one model, a block at a time. GPLv3. C11, libc + pthread.
# spel is built on the hako engine's trainer: HAKO is a checkout of
# github.com/mithraeums/hako, next to this one by default.
CC    ?= cc
HAKO  ?= ../hako
ARCH  ?= -march=native
FLAGS := -std=c11 -D_DEFAULT_SOURCE -Wall -Wextra -pthread -I$(HAKO)/src -I$(HAKO)/include
ENGINE := $(addprefix $(HAKO)/src/,train.c iloss.c nn.c quant.c loader.c bpe.c)
OURS  := src/chain.c src/sha256.c src/ed25519.c src/net.c
# GPU=1: training on the GPU through OpenCL, loaded at run time (the engine's src/gpu.h).
# Loss checks are the engine's integer pass (iloss.c) on the CPU either way, the same to the bit everywhere.
GPU_SRC  := $(HAKO)/src/gpu.c
GPU_LIBS := $(if $(filter Linux,$(shell uname -s)),-ldl,)
ifeq ($(GPU),1)
  ENGINE += $(GPU_SRC)
  FLAGS  += -DHAKM_GPU
  LIBS   := $(GPU_LIBS)
endif
SRC   := $(OURS) $(ENGINE)
HDR   := $(wildcard src/*.h $(HAKO)/src/*.h $(HAKO)/include/*.h)

# no -ffast-math: training checks the loss for NaN, and fast-math may assume there is none
spel: cli/spel.c $(SRC) $(HDR)
	@[ -f $(HAKO)/src/train.c ] || { echo "no hako engine at $(HAKO): git clone https://github.com/mithraeums/hako $(HAKO), or make HAKO=/path/to/hako"; exit 1; }
	$(CC) -O3 $(FLAGS) -ffp-contract=fast $(ARCH) -o $@ cli/spel.c $(SRC) -lm -lpthread $(LIBS)

B := build
SAN := $(shell printf 'int main(void){return 0;}\n' | $(CC) -fsanitize=address,undefined -x c - -o /dev/null 2>/dev/null && echo -fsanitize=address,undefined)
$(B):
	mkdir -p $(B)
$(B)/test_chain: tests/test_chain.c $(SRC) $(HDR) | $(B)
	$(CC) -O1 -g $(FLAGS) $(SAN) $(ARCH) -o $@ tests/test_chain.c $(SRC) -lm -lpthread

$(B)/test_keys: tests/test_keys.c src/ed25519.c src/ed25519.h | $(B)
	$(CC) -O1 -g $(FLAGS) $(SAN) -o $@ tests/test_keys.c src/ed25519.c

$(B)/test_net: tests/test_net.c $(SRC) $(HDR) | $(B)
	$(CC) -O1 -g $(FLAGS) $(SAN) $(ARCH) -o $@ tests/test_net.c $(SRC) -lm -lpthread

# built as files, run every time; then our own sources with warnings as errors
test: $(B)/test_keys $(B)/test_chain $(B)/test_net werror-check
	@[ -n "$(SAN)" ] || echo "test: this compiler has no ASan/UBSan, running the tests without them"
	@$(B)/test_keys && $(B)/test_chain && $(B)/test_net

werror-check:
	@for f in src/*.c cli/*.c tests/*.c web/*.c; do $(CC) -O2 $(FLAGS) -Werror -c $$f -o /dev/null || exit 1; done; \
	echo "werror-check: spel compiles clean with warnings as errors"

# the GPU build, whatever GPU= says, warnings as errors (it is not the default binary)
gpu-check: | $(B)
	@$(CC) -O2 $(filter-out -DHAKM_GPU,$(FLAGS)) -DHAKM_GPU -Werror $(ARCH) -o $(B)/spel-gpu cli/spel.c $(OURS) $(filter-out $(GPU_SRC),$(ENGINE)) $(GPU_SRC) -lm -lpthread $(GPU_LIBS) && \
	echo "gpu-check: spel builds for the GPU (make GPU=1) with warnings as errors"

# the arm64 build from an x86 Mac, and the other way round
cross-check:
	@for t in arm64-apple-macos11 x86_64-apple-macos11; do \
		printf 'int x;' | $(CC) -target $$t -x c -c - -o /dev/null 2>/dev/null || { echo "cross-check: this compiler cannot target $$t, skipped"; continue; }; \
		for f in src/*.c cli/*.c tests/*.c $(ENGINE); do $(CC) -target $$t -O2 $(FLAGS) -Werror -c $$f -o /dev/null || exit 1; done; done; \
	echo "cross-check: compiles for arm64 and x86_64"

# the page's trainer: the same C for wasm32-wasip1, from wasi-sdk (hako-studio's
# `make toolchain` fetches it to the same place). Single thread, SIMD on.
WASI_VER  ?= 33.0
WASI_DIR  ?= $(HOME)/.hako/toolchains/wasi-sdk-$(WASI_VER)-$(shell uname -m)-macos
CC_WASM   ?= $(WASI_DIR)/bin/clang
WASMFLAGS := --target=wasm32-wasip1 --sysroot=$(WASI_DIR)/share/wasi-sysroot -O3 -msimd128 -std=c11 -D_DEFAULT_SOURCE \
             -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_GETPID -I$(HAKO)/src -I$(HAKO)/include \
             -Wl,-z,stack-size=8388608 -Wl,--strip-all -lwasi-emulated-mman -lwasi-emulated-getpid
# the page's GPU is WebGPU (web/gpu_web.c on web/gpu.js), never the engine's OpenCL
WASMSRC   := web/spel.c web/gpu_web.c src/chain.c src/sha256.c src/ed25519.c $(filter-out $(GPU_SRC),$(ENGINE))
web/spel.wasm: $(WASMSRC) $(HDR)
	@[ -x "$(CC_WASM)" ] || { echo "no wasi-sdk at $(WASI_DIR): see hako-studio's make toolchain, or make CC_WASM=... WASI_DIR=..."; exit 1; }
	$(CC_WASM) $(WASMSRC) $(WASMFLAGS) -DHAKM_GPU -Wall -Wextra -o $@
wasm: web/spel.wasm
test-web: spel web/spel.wasm
	@[ -x $(HAKO)/hakm-train ] || $(MAKE) -C $(HAKO) hakm-train
	@command -v node >/dev/null || { echo "test-web: needs node"; exit 1; }
	@HAKO=$(HAKO) sh web/test.sh

# the page's GPU training (web/gpu.js) held to the CPU inside spel.wasm, under deno's WebGPU;
# skips, passing, with no deno or no GPU
test-gpu-web: web/spel.wasm
	@command -v deno >/dev/null || { echo "test-gpu-web: no deno, skipped"; exit 0; }; \
	deno run --unstable-webgpu --allow-read web/test-gpu.mjs

clean:
	rm -rf spel $(B) web/spel.wasm

.PHONY: test werror-check gpu-check cross-check clean wasm test-web test-gpu-web
