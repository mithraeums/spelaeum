# Spelaeum — block mining applied to training. GPLv3. C11, libc + pthread.
# spel is built on the hako engine's trainer: HAKO is a checkout of
# github.com/mithraeums/hako, next to this one by default.
CC    ?= cc
HAKO  ?= ../hako
ARCH  ?= -march=native
FLAGS := -std=c11 -D_DEFAULT_SOURCE -Wall -Wextra -pthread -I$(HAKO)/src -I$(HAKO)/include
ENGINE := $(addprefix $(HAKO)/src/,train.c nn.c quant.c loader.c bpe.c)
SRC   := src/chain.c src/sha256.c $(ENGINE)
HDR   := $(wildcard src/*.h $(HAKO)/src/*.h $(HAKO)/include/*.h)

# no -ffast-math: mining checks the loss for NaN, and fast-math may assume there is none
spel: cli/spel.c $(SRC) $(HDR)
	@[ -f $(HAKO)/src/train.c ] || { echo "no hako engine at $(HAKO): git clone https://github.com/mithraeums/hako $(HAKO), or make HAKO=/path/to/hako"; exit 1; }
	$(CC) -O3 $(FLAGS) -ffp-contract=fast $(ARCH) -o $@ cli/spel.c $(SRC) -lm -lpthread

B := build
SAN := $(shell printf 'int main(void){return 0;}\n' | $(CC) -fsanitize=address,undefined -x c - -o /dev/null 2>/dev/null && echo -fsanitize=address,undefined)
$(B):
	mkdir -p $(B)
$(B)/test_chain: tests/test_chain.c $(SRC) $(HDR) | $(B)
	$(CC) -O1 -g $(FLAGS) $(SAN) $(ARCH) -o $@ tests/test_chain.c $(SRC) -lm -lpthread

# built as a file, run every time; then our own sources with warnings as errors
test: $(B)/test_chain werror-check
	@[ -n "$(SAN)" ] || echo "test: this compiler has no ASan/UBSan, running the tests without them"
	@$(B)/test_chain

werror-check:
	@for f in src/*.c cli/*.c tests/*.c; do $(CC) -O2 $(FLAGS) -Werror -c $$f -o /dev/null || exit 1; done; \
	echo "werror-check: spel compiles clean with warnings as errors"

# the arm64 build from an x86 Mac, and the other way round
cross-check:
	@for t in arm64-apple-macos11 x86_64-apple-macos11; do \
		printf 'int x;' | $(CC) -target $$t -x c -c - -o /dev/null 2>/dev/null || { echo "cross-check: this compiler cannot target $$t, skipped"; continue; }; \
		for f in src/*.c cli/*.c tests/*.c $(ENGINE); do $(CC) -target $$t -O2 $(FLAGS) -Werror -c $$f -o /dev/null || exit 1; done; done; \
	echo "cross-check: compiles for arm64 and x86_64"

clean:
	rm -rf spel $(B)

.PHONY: test werror-check cross-check clean
