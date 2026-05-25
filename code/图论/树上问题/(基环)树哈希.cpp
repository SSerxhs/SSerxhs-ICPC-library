vector<int> tree_hash(const vector<vector<int>> &e, int root)//[0,n)
{
	int n = e.size();
	static map<vector<int>, int> mp;
	static int id = 0;
	vector<int> h(n), ed(n);
	function<void(int)> dfs = [&](int u) {
		ed[u] = 1;
		vector<int> c;
		for (int v : e[u]) if (!ed[v])
		{
			dfs(v);
			c.push_back(h[v]);
		}
		sort(all(c));
		if (!mp.count(c)) mp[c] = id++;
		h[u] = mp[c];
	};
	dfs(root);
	return h;
}
vector<int> tree_hash(const vector<vector<int>> &e)//[0,n)
{
	int n = e.size();
	if (n == 0) return { };
	vector<int> sz(n), mx(n);
	function<void(int)> dfs = [&](int u) {
		sz[u] = 1;
		for (int v : e[u]) if (!sz[v])
		{
			dfs(v);
			sz[u] += sz[v];
			cmax(mx[u], sz[v]);
		}
		cmax(mx[u], n - sz[u]);
	};
	dfs(0);
	int m = *min_element(all(mx)), i;
	vector<int> rt;
	for (i = 0; i < n; i++) if (mx[i] == m) rt.push_back(i);
	for (int &u : rt) u = tree_hash(e, u)[u];
	sort(all(rt));
	return rt;
}
template<class T> void min_order(vector<T> &a)
{
	int n = a.size(), i, j, k;
	a.resize(n * 2);
	for (i = 0; i < n; i++) a[i + n] = a[i];
	i = k = 0; j = 1;
	while (i < n && j < n && k < n)
	{
		T x = a[i + k], y = a[j + k];
		if (x == y) ++k; else
		{
			(x > y ? i : j) += k + 1;
			j += (i == j);
			k = 0;
		}
	}
	a.resize(n);
	//[min(i,j),n)+[0,min(i,j))
	rotate(a.begin(), min(i, j) + all(a));
}
vector<int> pseudotree_hash(const vector<vector<int>> &e)//[0,n)
{
	int n = e.size();
	static map<vector<int>, int> mp;
	static int id = 0;
	vector<int> f(n), ed(n), h(n);
	pair lp{-1, -1};
	function<void(int)> dfs = [&](int u) {
		ed[u] = 1;
		for (int v : e[u]) if (!ed[v])
		{
			f[v] = u;
			dfs(v);
		}
		else if (v != f[u]) lp = {u, v};
	};
	dfs(0);
	auto [x, y] = lp;
	vector<int> node = {y};
	do node.push_back(y = f[y]); while (y != x);
	fill(all(ed), 0);
	for (int u : node) ed[u] = 1;
	dfs = [&](int u) {
		ed[u] = 1;
		vector<int> c;
		for (int v : e[u]) if (!ed[v])
		{
			dfs(v);
			c.push_back(h[v]);
		}
		sort(all(c));
		if (!mp.count(c)) mp[c] = id++;
		h[u] = mp[c];
	};
	vector<int> r0;
	for (int u : node)
	{
		dfs(u);
		r0.push_back(h[u]);
	}
	auto r1 = r0;
	reverse(all(r1));
	min_order(r0);
	min_order(r1);
	return min(r0, r1);
}
