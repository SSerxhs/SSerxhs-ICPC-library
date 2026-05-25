#include"bits/stdc++.h"
using namespace std;
typedef long long ll;
typedef double db;
int read() {
	int res = 0;
	char c = getchar(), f = 1;
	while (c < 48 || c>57) { if (c == '-')f = 0; c = getchar(); }
	while (c >= 48 && c <= 57)res = (res << 3) + (res << 1) + (c & 15), c = getchar();
	return f ? res : -res;
}

const int L = 1 << 19, mod = 1e9 + 7;
const db pi2 = 3.141592653589793 * 2;
int inc(int x, int y) { return x + y >= mod ? x + y - mod : x + y; }
int dec(int x, int y) { return x - y < 0 ? x - y + mod : x - y; }
int mul(int x, int y) { return (ll)x * y % mod; }
int qpow(int x, int y) {
	int res = 1;
	for (; y; y >>= 1)res = y & 1 ? mul(res, x) : res, x = mul(x, x);
	return res;
}
int inv(int x) { return qpow(x, mod - 2); }

struct cp {
	db x, y;
	cp() { }
	cp(db a, db b) { x = a, y = b; }
	cp operator+(const cp &p)const { return cp(x + p.x, y + p.y); }
	cp operator-(const cp &p)const { return cp(x - p.x, y - p.y); }
	cp operator*(const cp &p)const { return cp(x * p.x - y * p.y, x * p.y + y * p.x); }
	cp conj() { return cp(x, -y); }
}w[L];
int re[L];
int getre(int n) {
	int len = 1, bit = 0;
	while (len < n)++bit, len <<= 1;
	for (int i = 1; i < len; ++i)re[i] = (re[i >> 1] >> 1) | ((i & 1) << (bit - 1));
	return len;
}
void getw() {
	for (int i = 0; i < L; ++i)w[i] = cp(cos(pi2 / L * i), sin(pi2 / L * i));
}
void fft(cp *a, int len, int m) {
	for (int i = 1; i < len; ++i)if (i < re[i])swap(a[i], a[re[i]]);
	for (int k = 1, r = L >> 1; k < len; k <<= 1, r >>= 1)
		for (int i = 0; i < len; i += k << 1)
			for (int j = 0; j < k; ++j) {
				cp &L = a[i + j], &R = a[i + j + k], t = w[r * j] * R;
				R = L - t, L = L + t;
			}
	if (!~m) {
		reverse(a + 1, a + len);
		cp tmp = cp(1.0 / len, 0);
		for (int i = 0; i < len; ++i)a[i] = a[i] * tmp;
	}
}
void mul(int *a, int *b, int *c, int n1, int n2, int n) {
	static cp f1[L], f2[L], f3[L], f4[L];
	int len = getre(n1 + n2 - 1);
	for (int i = 0; i < len; ++i) {
		f1[i] = i < n1 ? cp(a[i] >> 15, a[i] & 32767) : cp(0, 0);
		f2[i] = i < n2 ? cp(b[i] >> 15, b[i] & 32767) : cp(0, 0);
	}
	fft(f1, len, 1), fft(f2, len, 1);
	cp t1 = cp(0.5, 0), t2 = cp(0, -0.5), r = cp(0, 1);
	cp x1, x2, x3, x4;
	for (int i = 0; i < len; ++i) {
		int j = (len - i) & (len - 1);
		x1 = (f1[i] + f1[j].conj()) * t1;
		x2 = (f1[i] - f1[j].conj()) * t2;
		x3 = (f2[i] + f2[j].conj()) * t1;
		x4 = (f2[i] - f2[j].conj()) * t2;
		f3[i] = x1 * (x3 + x4 * r);
		f4[i] = x2 * (x3 + x4 * r);
	}
	fft(f3, len, -1), fft(f4, len, -1);
	ll c1, c2, c3, c4;
	for (int i = 0; i < n; ++i) {
		c1 = (ll)(f3[i].x + 0.5) % mod, c2 = (ll)(f3[i].y + 0.5) % mod;
		c3 = (ll)(f4[i].x + 0.5) % mod, c4 = (ll)(f4[i].y + 0.5) % mod;
		c[i] = ((((c1 << 15) + c2 + c3) << 15) + c4) % mod;
	}
}
void inv(int *a, int *b, int n) {
	if (n == 1) { b[0] = 1; return; }
	static int c[L];
	int l = (n + 1) >> 1;
	inv(a, b, l);
	mul(a, b, c, n, l, n);
	for (int i = 0; i < n; ++i)c[i] = mod - c[i];
	c[0] += 2;
	mul(b, c, b, n, n, n);
}
void der(int *a, int n) {
	for (int i = 1; i < n; ++i)a[i - 1] = mul(a[i], i);
	a[n - 1] = 0;
}
void its(int *a, int n) {
	for (int i = n - 1; i; --i)a[i] = mul(a[i - 1], inv(i));
	a[0] = 0;
}
void ln(int *a, int *b, int n) {
	static int c[L];
	for (int i = 0; i < n; ++i)c[i] = a[i];
	der(c, n);
	inv(a, b, n);
	mul(b, c, b, n, n, n);
	its(b, n);
}
void exp(int *a, int *b, int n) {
	if (n == 1) { b[0] = 1; return; }
	static int c[L];
	int l = (n + 1) >> 1;
	exp(a, b, l);
	ln(b, c, n);
	for (int i = 0; i < n; ++i)c[i] = dec(a[i], c[i]);
	++c[0];
	mul(b, c, b, l, n, n);
	for (int i = 0; i < n; ++i)c[i] = 0;
}

int n, k, a[L], f[L], g[L];
int main() {
	getw();
	n = read(), k = read();
	for (int i = 1; i <= k; ++i)a[i] = inv(i);
	for (int i = 2; i <= n; ++i)
		for (int j = 1; i * j <= k; ++j)
			f[i * j] = inc(f[i * j], a[j]);
	for (int i = 1; i <= k; ++i)f[i] = mod - f[i];
	for (int i = 1; i <= k; ++i)f[i] = inc(f[i], mul(n - 1, a[i]));
	exp(f, g, k + 1);
	printf("%d\n", g[k]);
}
