{
	static int OP = 0;
	if (OP++ == 0)
	{
		int size = 256 << 20; // 256MB
		char *p = (char *)malloc(size) + size;
		__asm__("movl %0, %%esp\n" :: "r"(p));
	}
}
{
	static int OP = 0;
	if (OP++ == 0)
	{
		int size = 128 << 20;//128MB
		char *p = new char[size] + size;
		__asm__ __volatile__("movq %0, %%rsp\n""pushq $exit\n""jmp main\n"::"r"(p));
	}
}

