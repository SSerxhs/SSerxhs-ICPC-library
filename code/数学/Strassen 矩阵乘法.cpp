#include "bits/stdc++.h"
using namespace std;
using ui = unsigned;
using ull = unsigned long long;
const ui p = 998244353;
const ull fh = 1ull << 31;
struct Q
{
	ui **a;
	int n;
	Q() { n = 0; }
	void clear()
	{
		for (int i = 0; i < n; i++) delete a[i];
		if (n) delete a; n = 0;
	}
	Q(int nn)//不能传入不是 2 的幂的数！
	{
		n = nn;
		assert(n == (n & -n));
		a = new ui * [n];
		for (int i = 0; i < n; i++) a[i] = new ui[n], memset(a[i], 0, n * sizeof a[0][0]);
	}
	const Q &operator=(const Q &b)
	{
		clear(); n = b.n;
		a = new ui * [n];
		for (int i = 0; i < n; i++) a[i] = new ui[n], memcpy(a[i], b.a[i], n * sizeof a[0][0]);
		return *this;
	}
	~Q() { clear(); }
	Q operator+(const Q &b)
	{
		Q c(n);
		for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if ((c.a[i][j] = a[i][j] + b.a[i][j]) >= p) c.a[i][j] -= p;
		return c;
	}
	Q operator-(const Q &b)
	{
		Q c(n);
		for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if ((c.a[i][j] = a[i][j] - b.a[i][j]) & fh) c.a[i][j] += p;
		return c;
	}
	Q operator*(Q &b)
	{
		Q c(n);
		if (n <= 128)
		{
			for (int i = 0; i < n; i++) for (int k = 0; k < n; k++) for (int j = 0; j < n; j++) c.a[i][j] = (c.a[i][j] + (ull)a[i][k] * b.a[k][j]) % p;
			return c;
		}
		Q A[2][2], B[2][2], s[10], p[5];
		n >>= 1;
		int i, j, k, l;
		for (i = 0; i < 2; i++) for (j = 0; j < 2; j++)
		{
			A[i][j] = Q(n);
			for (k = 0; k < n; k++) memcpy(A[i][j].a[k], a[k + i * n] + j * n, n * sizeof a[0][0]);
			B[i][j] = Q(n);
			for (k = 0; k < n; k++) memcpy(B[i][j].a[k], b.a[k + i * n] + j * n, n * sizeof a[0][0]);
		}
		s[0] = B[0][1] - B[1][1];
		s[1] = A[0][0] + A[0][1];
		s[2] = A[1][0] + A[1][1];
		s[3] = B[1][0] - B[0][0];
		s[4] = A[0][0] + A[1][1];
		s[5] = B[0][0] + B[1][1];
		s[6] = A[0][1] - A[1][1];
		s[7] = B[1][0] + B[1][1];
		s[8] = A[0][0] - A[1][0];
		s[9] = B[0][0] + B[0][1];
		p[0] = A[0][0] * s[0];
		p[1] = s[1] * B[1][1];
		p[2] = s[2] * B[0][0];
		p[3] = A[1][1] * s[3];
		p[4] = s[4] * s[5];
		A[0][0] = p[4] + p[3] - p[1] + s[6] * s[7];
		A[0][1] = p[0] + p[1];
		A[1][0] = p[2] + p[3];
		A[1][1] = p[4] + p[0] - p[2] - s[8] * s[9];
		for (i = 0; i < 2; i++) for (j = 0; j < 2; j++)	for (k = 0; k < n; k++) memcpy(c.a[k + i * n] + j * n, A[i][j].a[k], n * sizeof a[0][0]);
		n <<= 1;
		return c;
	}
};
int main()
{
	int i, j, n, m, k;
	ios::sync_with_stdio(0); cin.tie(0);
	cin >> n >> m >> k;
	int N = 1 << 32 - min({__builtin_clz(n - 1), __builtin_clz(m - 1), __builtin_clz(k - 1)});
	Q a(N), b(N);
	for (i = 0; i < n; i++) for (j = 0; j < m; j++) cin >> a.a[i][j];
	for (i = 0; i < m; i++) for (j = 0; j < k; j++) cin >> b.a[i][j];
	a = a * b;
	for (i = 0; i < n; i++) for (j = 0; j < k; j++) cout << a.a[i][j] << " \n"[j + 1 == k];
}

