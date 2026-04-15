class fast_iostream
{
private:
#define SIGNED
	const static int MAXBF = 1 << 20; FILE *inf, *ouf;
	char *inbuf, *inst, *ined;
	char *oubuf, *oust, *oued;
	inline void _flush() { fwrite(oubuf, 1, oued - oust, ouf); }
	inline char _getchar() {
		if (inst == ined) inst = inbuf, ined = inbuf + fread(inbuf, 1, MAXBF, inf);
		return inst == ined ? EOF : *inst++;
	}
	inline void _putchar(char c) {
		if (oued == oust + MAXBF) _flush(), oued = oubuf;
		*oued++ = c;
	}
public:
	fast_iostream(FILE *_inf = stdin, FILE *_ouf = stdout)
		:inbuf(new char[MAXBF]), inf(_inf), inst(inbuf), ined(inbuf),
		oubuf(new char[MAXBF]), ouf(_ouf), oust(oubuf), oued(oubuf) {
	}
	~fast_iostream() { _flush(); delete inbuf; delete oubuf; }
	fast_iostream &operator >> (char &c) {
		while (isspace(c = _getchar()));
		return *this;
	}
	fast_iostream &operator >> (string &s) {
		static char c;
		while (isspace(c = _getchar()));
		s = c;
		while (!isspace(c = _getchar())) s += c;
		return *this;
	}
	template <class Int>
	fast_iostream &operator >> (Int &n) {
		static char c;
#ifdef SIGNED
		bool neg = 0;
		while ((c = _getchar()) < '0' || c > '9') neg |= c == '-';
#else
		while ((c = _getchar()) < '0' || c > '9');
#endif
		n = c - '0';
		while ((c = _getchar()) >= '0' && c <= '9') n = n * 10 + c - '0';
#ifdef SIGNED
		if (neg) n = -n;
#endif
		return *this;
	}
	template <class Int>
	fast_iostream &operator << (Int   n) {
		if (n < 0) _putchar('-'), n = -n; static char S[20]; int t = 0;
		do { S[t++] = '0' + n % 10, n /= 10; } while (n);
		for (int i = 0; i < t; ++i) _putchar(S[t - i - 1]);
		return *this;
	}
	fast_iostream &operator << (char  c) { _putchar(c);    return *this; }
	fast_iostream &operator << (const char *s) {
		for (int i = 0; s[i]; ++i) _putchar(s[i]); return *this;
	}
}fio;
