namespace KM
{
	const int N = 405;//点数
	typedef long long ll;//答案范围
	const ll inf = 1e16;
	int lk[N], kl[N], pre[N], q[N], n, h, t;
	ll sl[N], e[N][N], lx[N], ly[N];
	bool edx[N], edy[N];
	bool ck(int v)
	{
		if (edy[v] = 1, kl[v]) return edx[q[++t] = kl[v]] = 1;
		while (v) swap(v, lk[kl[v] = pre[v]]);
		return 0;
	}
	void bfs(int u)
	{
		fill_n(sl + 1, n, inf);
		memset(edx + 1, 0, n * sizeof edx[0]);
		memset(edy + 1, 0, n * sizeof edy[0]);
		q[h = t = 1] = u; edx[u] = 1;
		while (1)
		{
			while (h <= t)
			{
				int u = q[h++], v;
				ll d;
				for (v = 1; v <= n; v++) if (!edy[v] && sl[v] >= (d = lx[u] + ly[v] - e[u][v])) if (pre[v] = u, d) sl[v] = d; else if (!ck(v)) return;
			}
			int i;
			ll m = inf;
			for (i = 1; i <= n; i++) if (!edy[i]) m = min(m, sl[i]);
			for (i = 1; i <= n; i++)
			{
				if (edx[i]) lx[i] -= m;
				if (edy[i]) ly[i] += m; else sl[i] -= m;
			}
			for (i = 1; i <= n; i++) if (!edy[i] && !sl[i] && !ck(i)) return;
		}
	}
	template<class TT> ll max_weighted_match(int N, const vector<tuple<int, int, TT>> &edges)//lk[[1,n]]->[1,n]
	{
		int i; n = N;
		memset(lk + 1, 0, n * sizeof lk[0]);
		memset(kl + 1, 0, n * sizeof kl[0]);
		memset(ly + 1, 0, n * sizeof ly[0]);
		for (i = 1; i <= n; i++) fill_n(e[i] + 1, n, 0); // 若不需保证匹配边最多，置 0 即可，否则 -inf/N
		for (auto [u, v, w] : edges) e[u][v] = max(e[u][v], (ll)w);
		for (i = 1; i <= n; i++) lx[i] = *max_element(e[i] + 1, e[i] + n + 1);
		for (i = 1; i <= n; i++) bfs(i);
		ll r = 0;
        for (i = 1; i <= n; i++)
            if (!e[i][lk[i]]) // 若不需保证匹配边最多，需要去掉这些辅助边
                kl[lk[i]] = 0, lk[i] = 0;
            else
                r += e[i][lk[i]];
		return r;
	}
}
using KM::max_weighted_match, KM::lk, KM::kl, KM::e;

