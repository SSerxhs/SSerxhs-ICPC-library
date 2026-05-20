vector<int> bipolar_orientation(const vector<pair<int, int>> &edges, int n, int s, int t)//[0,n)
{
	assert(s != t);
	vector e(n, vector<int>());
	for (auto [u, v] : edges)
	{
		e[u].push_back(v);
		e[v].push_back(u);
	}
	int cur = 1, i;
	vector<int> pre(n), low(n), p(n);
	pre[s] = 1;
	vector<int> id;
	bool flg = 0;
	function<void(int)> dfs = [&](int x) {
		pre[x] = ++cur;
		low[x] = x;
		for (int y : e[x])
		{
			flg |= y == s;
			if (pre[y] == 0)
			{
				id.push_back(y);
				dfs(y);
				p[y] = x;
				if (pre[low[y]] < pre[low[x]]) low[x] = low[y];
			}
			else if (pre[y] != 0 && pre[y] < pre[low[x]]) low[x] = y;
		}
	};
	dfs(t);
	if (!flg) return { };
	vector<int> sign(n, -1);
	vector<int> l(n), r(n);
	r[s] = t;
	l[t] = s;
	for (int v : id)
	{
		if (sign[low[v]] == -1)
		{
			l[v] = l[p[v]];
			r[l[v]] = v;
			l[p[v]] = v;
			r[v] = p[v];
			sign[p[v]] = 1;
		}
		else
		{
			r[v] = r[p[v]];
			l[r[v]] = v;
			r[p[v]] = v;
			l[v] = p[v];
			sign[p[v]] = -1;
		}
	}
	vector<int> a(n);
	int x;
	for (i = 0, x = s; i < n; x = r[x], i++) a[i] = x;
	vector<int> ia(n, -1), rd(n), cd(n);
	for (i = 0; i < n; i++) ia[a[i]] = i;
	if (count(all(ia), -1)) return { };
	for (auto [u, v] : edges)
	{
		if (ia[u] > ia[v]) swap(u, v);
		++cd[u]; ++rd[v];
	}
	for (i = 0; i < n; i++) if (i != s && i != t && (!cd[i] || !rd[i])) return { };
	return a;
}


