ull det(vector<vector<ull>> b)
{
	ull r = 1;
	int n = b.size(), i, j, k;
	for (i = 0; i < n; i++)
	{
		for (j = i; j < n; j++) if (b[j][i]) break;
		if (j == n) return 0;
		swap(b[j], b[i]);
		if (j != i) r = (p - r) % p;
		r = r * b[i][i] % p;
		b[i][i] = ksm(b[i][i], p - 2);
		for (j = n - 1; j >= i; j--) b[i][j] = b[i][j] * b[i][i] % p;
		for (j = i + 1; j < n; j++) for (k = n - 1; k >= i; k--) b[j][k] = (b[j][k] + (p - b[j][i]) * b[i][k]) % p;
	}
	return r;
}
ull euler_path_count(vector<vector<int>> a, int s, int t)
{
	int n = a.size(), i, j, k;
	++a[t][s]; s = t;
	vector<int> rd(n), cd(n);
	for (i = 0; i < n; i++) for (j = 0; j < n; j++) cd[i] += a[i][j], rd[j] += a[i][j];
	for (i = 0; i < n; i++) if (cd[i] != rd[i]) return 0;
	vector<int> f(n);
	iota(all(f), 0);
	function<int(int)> getf = [&](int u) { return f[u] == u ? u : f[u] = getf(f[u]); };
	for (i = 0; i < n; i++) for (j = 0; j < n; j++) if (a[i][j]) f[getf(i)] = getf(j);
	ull r = 1;
	vector<int> id;
	for (i = 0; i < n; i++) if (cd[i])
	{
		if (getf(i) != getf(s)) return 0;
		r = r * fac[cd[i] - 1] % p;
		if (i != s) id.push_back(i);
	}
	n = id.size();
	vector b(n, vector<ull>(n));
	for (i = 0; i < n; i++)
	{
		b[i][i] = cd[id[i]] - a[id[i]][id[i]];
		for (j = 0; j < n; j++) if (i != j) b[i][j] = (p - a[id[i]][id[j]]) % p;
	}
	return r * det(b) % p;
}
ull euler_path_count(vector<vector<int>> a)
{
	int n = a.size(), i, j, s = -1, t = -1;
	vector<int> rd(n), cd(n), d(n);
	for (i = 0; i < n; i++) for (j = 0; j < n; j++) cd[i] += a[i][j], rd[j] += a[i][j];
	if (count(all(cd), 0) == n) return 1;
	for (i = 0; i < n; i++) d[i] = cd[i] - rd[i];
	s = max_element(all(d)) - d.begin();
	t = min_element(all(d)) - d.begin();
	ull r = 0;
	if (s == t)
	{
		for (i = 0; i < n; i++) if (cd[i]) r += euler_path_count(a, i, i);
	}
	else r = euler_path_count(a, s, t);
	return r % p;
}
ull euler_circuit_count(vector<vector<int>> a)
{
	int n = a.size(), i, j;
	for (i = 0; i < n; i++) for (j = 0; j < n; j++) if (a[i][j]) return euler_path_count(a, i, i) * ksm(accumulate(all(a[i]), 0llu) % p, p - 2) % p;
	return 1;
}
ull directed_spanning_tree_count(vector<vector<int>> a, int s)
{
	int n = a.size(), i, j;
	vector b(n - 1, vector<ull>(n - 1));
	for (i = 0; i < n; i++) a[i][i] = 0;
	for (i = 0; i < n; i++) if (i != s) for (j = 0; j < n; j++) if (j != s && i != j) b[i - (i > s)][j - (j > s)] = (p - a[i][j]) % p;
	for (i = 0; i < n; i++) if (i != s) for (j = 0; j < n; j++) (b[i - (i > s)][i - (i > s)] += a[j][i]) %= p;
	return det(b);
}//外向
ull undirected_spanning_tree_count(vector<vector<int>> a)
{
	int n = a.size(), i, j;
	--n;
	vector b(n, vector<ull>(n));
	for (i = 0; i < n; i++) a[i][i] = 0;
	for (i = 0; i < n; i++) for (j = 0; j < n; j++) if (i != j) b[i][j] = (p - a[i][j]) % p;
	for (i = 0; i < n; i++) b[i][i] = reduce(all(a[i]), 0llu) % p;
	return det(b);
}
