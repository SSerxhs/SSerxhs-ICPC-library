%:%.cpp %.in
	g++ $< -o $@ -std=c++20 -DLOCAL -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -fsanitize=undefined
	./$@ < $@.in
