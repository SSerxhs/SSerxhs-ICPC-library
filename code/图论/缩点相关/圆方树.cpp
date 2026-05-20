#include "bits/stdc++.h"
using namespace std;
#if !defined(ONLINE_JUDGE)&&defined(LOCAL)
#include "my_header\debug.h"
#else
#define dbg(...); 1;
#endif
typedef unsigned int ui;
typedef long long ll;
#define all(x) (x).begin(),(x).end()
const int N = 3e4 + 2, M = 3e4 + 2;//M 包括方点
struct P
{
	int v, w, id;
	P(int a, int b, int c) :v(a), w(b), id(c) { }
};
struct Q
{
	int v, w;
	Q(int a, int b) :v(a), w(b) { }
};
vector<P> e[N];
vector<Q> fe[M];
int dfn[M], low[N], st[N], len[M], top[M], siz[M], hc[M], dep[M], f[M], rb[N];
bool ed[M];//ed,dfn,loop,sum,fe,hc,tp,id,cnt,dep[1] 需初始化（注意倍率），ed 大小为边数
int tp, id, cnt, n;
void dfs1(int u)
{
	dfn[u] = low[u] = ++id;
	st[++tp] = u;
	for (auto [v, w, id] : e[u]) if (!ed[id])
	{
		if (dfn[v]) low[u] = min(low[u], dfn[v]), rb[v] = w; else
		{
			ed[id] = 1;
			dfs1(v);
			if (dfn[u] > low[v]) low[u] = min(low[u], low[v]), rb[v] = w; else
			{
				int ntp = tp;
				while (st[ntp] != v) --ntp;
				if (ntp == tp)//圆圆边
				{
					--tp;
					fe[u].emplace_back(v, w);
					f[v] = u;
					continue;
				}
				++cnt; f[cnt] = u;
				for (int i = ntp; i <= tp; i++) f[st[i]] = cnt;
				len[st[ntp]] = w;
				for (int i = ntp + 1; i <= tp; i++) len[st[i]] = len[st[i - 1]] + rb[st[i]];
				len[cnt] = len[st[tp]] + rb[u];
				fe[u].emplace_back(cnt, 0);
				for (int i = ntp; i <= tp; i++) fe[cnt].emplace_back(st[i], min(len[st[i]], len[cnt] - len[st[i]]));
				tp = ntp - 1;
			}
		}
	}
}
void dfs2(int u)
{
	siz[u] = 1;
	for (auto [v, w] : fe[u])
	{
		dep[v] = dep[u] + w;
		dfs2(v);
		siz[u] += siz[v];
		if (siz[v] > siz[hc[u]]) hc[u] = v;
	}
}
void dfs3(int u)
{
	dfn[u] = ++id;
	if (hc[u])
	{
		top[hc[u]] = top[u];
		dfs3(hc[u]);
		for (auto [v, w] : fe[u]) if (v != hc[u]) dfs3(top[v] = v);
	}
}
int lca(int u, int v)
{
	while (top[u] != top[v]) if (dfn[top[u]] > dfn[top[v]]) u = f[top[u]]; else v = f[top[v]];//注意不能用 dep
	return dfn[u] < dfn[v] ? u : v;
}
int find(int u, int v)//u 是根
{
	if (dfn[hc[u]] + siz[hc[u]] > dfn[v]) return hc[u];
	while (f[top[v]] != u) v = f[top[v]];
	return top[v];
}
int dis(int u, int v)
{
	int o = lca(u, v), r = dep[u] + dep[v];
	if (o <= n) return r - (dep[o] << 1);
	u = find(o, u); v = find(o, v);
	if (len[u] > len[v]) swap(u, v);
	return r + min(len[v] - len[u], len[o] - (len[v] - len[u])) - dep[u] - dep[v];
}
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	int m, q, i;
	cin >> n >> m >> q; cnt = n;
	for (i = 1; i <= m; i++)
	{
		int u, v, w;
		cin >> u >> v >> w;
		e[u].emplace_back(v, w, i);
		e[v].emplace_back(u, w, i);
	}
	mt19937 rnd(time(0));
	for (i = 1; i <= n; i++) shuffle(all(e[i]), rnd);
	dfs1(1); id = 0;
	dfs2(1);
	dfs3(top[1] = 1);
	while (q--)
	{
		int u, v;
		cin >> u >> v;
		cout << dis(u, v) << '\n';
	}
}

