#include "bits/stdc++.h"
using namespace std;
typedef long long ll;
const int N = 1e6 + 2;
int ss[N], mn[N], fmn[N], phi[N];
int t, n, gs, i, d;
bool ed[N], av[N], yg[N], hv[N];
double inv[N];
void getfac(int x, int *a, int &n)
{
	int y = x, z;
	if (1 ^ x & 1)
	{
		a[n = 1] = 2; x >>= 1; while (1 ^ x & 1) x >>= 1;
	}
	while (x > 1)
	{
		x = 1e-9 + (x * inv[a[++n] = z = mn[x]]);
		while (x % z == 0) x = 1e-9 + x * inv[z];
	}
	for (i = 1; i <= n; i++) av[a[i]] = 0, a[i] = 1e-9 + (y * inv[a[i]]);
}
int ksm(int x, int y, int p)
{
	int r = 1;
	while (y)
	{
		if (y & 1) r = (ll)r * x % p;
		x = (ll)x * x % p; y >>= 1;
	}
	return r;
}
bool ck(int x, int *a, int n, int p)
{
	for (int i = 1; i <= n; i++) if (ksm(x, a[i], p) == 1) return 0;
	return 1;
}
void getrt(int x, int d)
{
	if (!hv[x]) return puts("0\n"), void();
	static int a[30];
	int n = 0, y, i, g = 0, c = d; y = phi[x];
	fill(av + 1, av + y + 1, 1);
	getfac(y, a, n);
	for (i = 1; i < x; i++) if (__gcd(i, x) == 1 && ck(i, a, n, x)) break;
	yg[g = i] = 1;//g就是最小原根
	int j = (ll)g * g % x;
	for (i = 2; i < y; i++, j = (ll)j * g % x) yg[j] = av[i] = av[mn[i]] & av[fmn[i]];
	printf("%d\n", phi[y]);
	for (i = 1; i < x; i++) if (yg[i])
	{
		yg[i] = 0;
		if (--c == 0) printf("%d ", i), c = d;
	}puts("");
}
void init()
{
	int i, j, k, n = N - 1;
	mn[1] = phi[1] = 1;
	for (i = 1; i <= n; i++) inv[i] = 1.0 / i;
	for (i = 2; i <= n; i++)
	{
		if (!ed[i]) phi[mn[i] = ss[++gs] = i] = i - 1, hv[i] = 1;
		for (j = 1; j <= gs && (k = ss[j] * i) <= n; j++)
		{
			ed[k] = 1; mn[k] = ss[j];
			if (i % ss[j] == 0) { phi[k] = phi[i] * ss[j]; hv[k] = hv[i]; break; }
			phi[k] = phi[i] * (ss[j] - 1);
		}
	}
	for (i = n; i; i--) fmn[i] = 1e-9 + (i * inv[mn[i]]), hv[i] |= (1 ^ i & 1) && hv[i >> 1];
	for (i = 8; i <= n; i <<= 1) hv[i] = 0;
}
int main()
{
	init();
	scanf("%d", &t);
	while (t--)
	{
		scanf("%d%d", &n, &d);
		getrt(n, d);
	}
}

