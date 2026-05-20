int siz[N], dep[N];
int n, ksiz, md, rt, mn;
bool ed[N];
void find(int u)
{
	ed[u] = 1; siz[u] = 1;
	int mx = 0;
	for (int v : e[u]) if (!ed[v])
	{
		find(v);
		siz[u] += siz[v];
		mx = max(mx, siz[v]);
	}
	mx = max(mx, ksiz - siz[u]);
	if (mn > mx) mn = mx, rt = u;
	ed[u] = 0;
}
void cal(int u)
{
	md = max(md, dep[u]);
	ed[u] = 1; ++cnt[dep[u]];
	for (int v : e[u]) if (!ed[v])
	{
		dep[v] = dep[u] + 1;
		cal(v);
	}
	ed[u] = 0;
}
void solve(int u)
{
	mn = 1e9;
	find(u);
	ed[rt] = 1;
	vector<int> c;
	for (int v : e[rt]) if (!ed[v])
	{
		c.push_back(v);
		if (siz[v] >= siz[rt]) siz[v] = siz[u] - siz[rt];
	}
	sort(all(c), [&](const int &a, const int &b) {return siz[a] < siz[b]; });
	NTT::Q a(vector<ui>{1});
	NT::Q b(vector<ui>{1});
	for (int v : c)
	{
		md = 0; dep[v] = 1;
		cal(v); ++md;
		vector<ui> d(cnt, cnt + md);
		NTT::Q e(d);
		NT::Q f(d);
		auto g = e & a;
		auto h = f & b;
		for (int i = 0; i < g.a.size(); i++) r1[i] = (r1[i] + g.a[i]) % NTT::p;
		for (int i = 0; i < h.a.size(); i++) r2[i] = (r2[i] + h.a[i]) % NT::p;
		a += e; b += f;
		fill_n(cnt, md, 0);
	}
	for (int v : c)
	{
		ksiz = siz[v];
		solve(v);
	}
}
