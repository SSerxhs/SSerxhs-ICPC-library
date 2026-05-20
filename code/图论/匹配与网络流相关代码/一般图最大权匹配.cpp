
namespace weighted_blossom_tree
{
#define d(x) (lab[x.u]+lab[x.v]-e[x.u][x.v].w*2)
	const int N = 403 * 2;//两倍点数
	typedef long long ll;//总和大小
	typedef int T;//权值大小
	//均不允许无符号
	const T inf = numeric_limits<int>::max() >> 1;
	struct Q
	{
		int u, v;
		T w;
	} e[N][N];
	T lab[N];
	int n, m = 0, id, h, t, lk[N], sl[N], st[N], f[N], b[N][N], s[N], ed[N], q[N];
	vector<int> p[N];
	void upd(int u, int v) { if (!sl[v] || d(e[u][v]) < d(e[sl[v]][v])) sl[v] = u; }
	void ss(int v)
	{
		sl[v] = 0;
		for (int u = 1; u <= n; u++) if (e[u][v].w > 0 && st[u] != v && !s[st[u]]) upd(u, v);
	}
	void ins(int u) { if (u <= n) q[++t] = u; else for (int v : p[u]) ins(v); }
	void mdf(int u, int w)
	{
		st[u] = w;
		if (u > n) for (int v : p[u]) mdf(v, w);
	}
	int gr(int u, int v)
	{
		if ((v = find(all(p[u]), v) - p[u].begin()) & 1)
		{
			reverse(1 + all(p[u]));
			return (int)p[u].size() - v;
		}
		return v;
	}
	void stm(int u, int v)
	{
		lk[u] = e[u][v].v;
		if (u <= n) return;
		Q w = e[u][v];
		int x = b[u][w.u], y = gr(u, x), i;
		for (i = 0; i < y; i++) stm(p[u][i], p[u][i ^ 1]);
		stm(x, v);
		rotate(p[u].begin(), y + all(p[u]));
	}
	void aug(int u, int v)
	{
		int w = st[lk[u]];
		stm(u, v);
		if (!w) return;
		stm(w, st[f[w]]);
		aug(st[f[w]], w);
	}
	int lca(int u, int v)
	{
		for (++id; u | v; swap(u, v))
		{
			if (!u) continue;
			if (ed[u] == id) return u;
			ed[u] = id;//??????????v?? 这是原作者的注释，我也不知道是啥
			if (u = st[lk[u]]) u = st[f[u]];
		}
		return 0;
	}
	void add(int u, int a, int v)
	{
		int x = n + 1, i, j;
		while (x <= m && st[x]) ++x;
		if (x > m) ++m;
		lab[x] = s[x] = st[x] = 0; lk[x] = lk[a];
		p[x].clear(); p[x].push_back(a);
		for (i = u; i != a; i = st[f[j]]) p[x].push_back(i), p[x].push_back(j = st[lk[i]]), ins(j);//复制，改一处
		reverse(1 + all(p[x]));
		for (i = v; i != a; i = st[f[j]]) p[x].push_back(i), p[x].push_back(j = st[lk[i]]), ins(j);
		mdf(x, x);
		for (i = 1; i <= m; i++) e[x][i].w = e[i][x].w = 0;
		memset(b[x] + 1, 0, n * sizeof b[0][0]);
		for (int u : p[x])
		{
			for (v = 1; v <= m; v++) if (!e[x][v].w || d(e[u][v]) < d(e[x][v])) e[x][v] = e[u][v], e[v][x] = e[v][u];
			for (v = 1; v <= n; v++) if (b[u][v]) b[x][v] = u;
		}
		ss(x);
	}
	void ex(int u)  // s[u] == 1
	{
		for (int x : p[u]) mdf(x, x);
		int a = b[u][e[u][f[u]].u], r = gr(u, a), i;
		for (i = 0; i < r; i += 2)
		{
			int x = p[u][i], y = p[u][i + 1];
			f[x] = e[y][x].u;
			s[x] = 1; s[y] = 0;
			sl[x] = 0; ss(y);
			ins(y);
		}
		s[a] = 1; f[a] = f[u];
		for (i = r + 1; i < p[u].size(); i++) s[p[u][i]] = -1, ss(p[u][i]);
		st[u] = 0;
	}
	bool on(const Q &e)
	{
		int u = st[e.u], v = st[e.v], a;
		if (s[v] == -1)
		{
			f[v] = e.u; s[v] = 1;
			a = st[lk[v]];
			sl[v] = sl[a] = s[a] = 0;
			ins(a);
		}
		else if (!s[v])
		{
			a = lca(u, v);
			if (!a) return aug(u, v), aug(v, u), 1;
			else add(u, a, v);
		}
		return 0;
	}
	bool bfs()
	{
		memset(s + 1, -1, m * sizeof s[0]);
		memset(sl + 1, 0, m * sizeof sl[0]);
		h = 1; t = 0;
		int i, j;
		for (i = 1; i <= m; i++) if (st[i] == i && !lk[i]) f[i] = s[i] = 0, ins(i);
		if (h > t) return 0;
		while (1)
		{
			while (h <= t)
			{
				int u = q[h++], v;
				if (s[st[u]] != 1) for (v = 1; v <= n; v++) if (e[u][v].w > 0 && st[u] != st[v])
				{
					if (d(e[u][v])) upd(u, st[v]); else if (on(e[u][v])) return 1;
				}
			}
			T x = inf;
			for (i = n + 1; i <= m; i++) if (st[i] == i && s[i] == 1) x = min(x, lab[i] >> 1);
			for (i = 1; i <= m; i++) if (st[i] == i && sl[i] && s[i] != 1) x = min(x, d(e[sl[i]][i]) >> s[i] + 1);
			for (i = 1; i <= n; i++) if (~s[st[i]]) if ((lab[i] += (s[st[i]] * 2 - 1) * x) <= 0) return 0;
			for (i = n + 1; i <= m; i++) if (st[i] == i && ~s[st[i]]) lab[i] += (2 - s[st[i]] * 4) * x;
			h = 1; t = 0;
			for (i = 1; i <= m; i++) if (st[i] == i && sl[i] && st[sl[i]] != i && !d(e[sl[i]][i]) && on(e[sl[i]][i])) return 1;
			for (i = n + 1; i <= m; i++) if (st[i] == i && s[i] == 1 && !lab[i]) ex(i);
		}
		return 0;
	}
	template<class TT> ll max_weighted_general_match(int N, const vector<tuple<int, int, TT>> &edges)//[1,n]，返回权值
	{
		memset(ed + 1, 0, m * sizeof ed[0]);
		memset(lk + 1, 0, m * sizeof lk[0]);
		n = m = N; id = 0;
		iota(st + 1, st + n + 1, 1);
		int i, j;
		T wm = 0;
		ll r = 0;
		for (i = 1; i <= n; i++) for (j = 1; j <= n; j++) e[i][j] = {i, j, 0};
		for (auto [u, v, w] : edges) wm = max(wm, e[v][u].w = e[u][v].w = max(e[u][v].w, (T)w));
		for (i = 1; i <= n; i++) p[i].clear();
		for (i = 1; i <= n; i++) for (j = 1; j <= n; j++) b[i][j] = i * (i == j);
		fill_n(lab + 1, n, wm);
		while (bfs());
		for (i = 1; i <= n; i++) if (lk[i]) r += e[i][lk[i]].w;
		return r / 2;
	}
#undef d
}
using weighted_blossom_tree::max_weighted_general_match, weighted_blossom_tree::lk;
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	int n, m;
	cin >> n >> m;
	vector<tuple<int, int, long long>> edges(m);
	for (auto &[u, v, w] : edges) cin >> u >> v >> w;
	cout << max_weighted_general_match(n, edges) << '\n';
	for (int i = 1; i <= n; i++) cout << lk[i] << " \n"[i == n];
}

