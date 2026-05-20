#include "bits/stdc++.h"
using namespace std;
typedef long long ll;
typedef pair<int, int> pa;
const int N = 1e6 + 2, M = 1e6 + 2;
ll ans;
pa w[N];
int c[N][2], f[N], fa[N], v[N], s[N], lz[N], lj[M], nxt[M], siz[N], hc[N], fir[N], st[N];
int a[N], top[N];
int n, i, x, y, z, bs, tp, rt;
void add()
{
	lj[++bs] = y; nxt[bs] = fir[x]; fir[x] = bs;
	lj[++bs] = x; nxt[bs] = fir[y]; fir[y] = bs;
}
void pushup(int &x)
{
	s[x] = min(v[x], min(s[c[x][0]], s[c[x][1]]));
}
void pushdown(int &x)
{
	if (lz[x] < 0)
	{
		int cc = c[x][0];
		if (cc)
		{
			lz[cc] += lz[x]; s[cc] += lz[x]; v[cc] += lz[x];
		}
		cc = c[x][1];
		if (cc)
		{
			v[cc] += lz[x]; lz[cc] += lz[x]; s[cc] += lz[x];
		}lz[x] = 0;
		return;
	}
}
bool nroot(int &x) { return c[f[x]][0] == x || c[f[x]][1] == x; }
bool cmp(pa &o, pa &p) { return o > p; }
void dfs1(int x)
{
	siz[x] = 1;
	for (int i = fir[x]; i; i = nxt[i]) if (lj[i] != fa[x])
	{
		fa[lj[i]] = x; dfs1(lj[i]); siz[x] += siz[lj[i]];
		if (siz[hc[x]] < siz[lj[i]]) hc[x] = lj[i];
	}
}
int build(int l, int r)
{
	if (l > r) return 0;
	if (l == r)
	{
		l = st[l]; s[l] = v[l] = siz[l] >> 1;
		return l;
	}
	int x = lower_bound(a + l, a + r + 1, a[l] + a[r] >> 1) - a, y = st[x];
	c[y][0] = build(l, x - 1);
	c[y][1] = build(x + 1, r);
	v[y] = siz[y] >> 1;
	if (c[y][0]) f[c[y][0]] = y;
	if (c[y][1]) f[c[y][1]] = y;
	pushup(y);
	return y;
}
void dfs2(int x)
{
	if (!hc[x]) return;
	int i;
	top[hc[x]] = top[x];
	if (top[x] == x)
	{
		st[tp = 1] = x;
		for (i = hc[x]; i; i = hc[i]) st[++tp] = i;
		for (i = 1; i <= tp; i++) a[i] = siz[st[i]] - siz[hc[st[i]]] + a[i - 1];
		f[build(1, tp)] = fa[x];
	}
	dfs2(hc[x]);
	for (i = fir[x]; i; i = nxt[i]) if (lj[i] != fa[x] && lj[i] != hc[x]) dfs2(top[lj[i]] = lj[i]);
}
void mdf(int x)
{
	int y = x;
	st[tp = 1] = x;
	while (y = f[y]) st[++tp] = y; y = x;
	while (tp) pushdown(st[tp--]);
	while (x)
	{
		--v[x]; --lz[c[x][0]]; --v[c[x][0]]; --s[c[x][0]];
		while (c[f[x]][0] == x) x = f[x]; x = f[x];
	}
	pushup(y);
	while (y = f[y]) pushup(y);
}
int ask(int x)
{
	int y = x;
	st[tp = 1] = x;
	while (y = f[y]) st[++tp] = y;
	while (tp) pushdown(st[tp--]);
	int r = v[x];
	while (x)
	{
		r = min(r, min(v[x], s[c[x][0]]));
		while (c[f[x]][0] == x) x = f[x]; x = f[x];
	}
	return r;
}
signed main()
{
	cin >> n; s[0] = 1e9;
	for (i = 1; i <= n; i++) cin >> w[w[i].second = i].first;
	for (i = 1; i < n; i++) cin >> x >> y, add();
	sort(w + 1, w + n + 1, cmp); dfs1(1); dfs2(top[1] = 1); rt = 1; while (f[rt]) rt = f[rt];
	for (i = 1; i <= n && v[rt]; i++) if (ask(w[i].second)) mdf(w[i].second), ans += w[i].first;
	cout << ans << endl;
}
