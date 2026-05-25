#include "bits/stdc++.h"
using namespace std;
const int N = 34;
struct Q
{
	int v, w, c;
	Q() { }
	Q(int x, int y, int z) :v(x), w(y), c(z) { }
};
vector<Q> lj[N];
int dis[N], cnt[N], pt[N], S;
Q pre[N], st[N];
int n, m, ans, tp;
bool ed[N];
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cin >> n >> m;
	while (m--)
	{
		int x, y, z, w;
		cin >> x >> y >> z >> w;
		lj[x].emplace_back(y, w, z);
		lj[y].emplace_back(x, 0, -z);
	}
	for (int i = 1; i <= n; i++) lj[0].emplace_back(i, 1, 0);
	while (1)
	{
		memset(dis, -0x3f, sizeof dis); dis[0] = 0;
		for (int i = 0; i <= n; i++) ed[i] = cnt[i] = 0; S = -1;
		queue<int> q; q.push(0);
		while (!q.empty())
		{
			int u = q.front(); q.pop(); ed[u] = 0;
			for (auto &[v, w, c] : lj[u]) if (w && dis[v] < dis[u] + c)
			{
				dis[v] = dis[u] + c; pre[v] = Q(u, w, c);
				if (!ed[v])
				{
					if (++cnt[v] > n + 1) { S = v; goto aa; }
					ed[v] = 1; q.push(v);
				}
			}
		}
	aa:;
		if (S == -1) break;
		{
			static bool ed[N];
			memset(ed, 0, sizeof ed);
			while (!ed[S]) ed[S] = 1, S = pre[S].v;
		}
		st[tp = 1] = pre[S]; pt[1] = S;
		int x = pre[S].v;
		while (x != S)
		{
			st[++tp] = pre[x]; pt[tp] = x;
			x = pre[x].v;
			assert(tp <= n + 5);
		}
		int fl = 1e9;
		for (int j = 1; j <= tp; j++) fl = min(fl, st[j].w);
		assert(fl);
		for (int j = 1; j <= tp; j++)
		{
			ans += fl * st[j].c;
			int nn = 0;
			for (auto &[v, w, c] : lj[st[j].v]) if (v == pt[j] && st[j].c == c && st[j].w == w) { ++nn; w -= fl; break; }
			for (auto &[v, w, c] : lj[pt[j]]) if (v == st[j].v && st[j].c + c == 0) { ++nn; w += fl; break; }assert(nn == 2);
		}
	}
	cout << ans << endl;
}
