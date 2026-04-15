struct SA
{
	int n;
	vector<vector<int>> st;
	vector<int> sa, rk, h;
	int lcp(int x, int y)
	{
		if (x == y) return n - x;
		x = rk[x]; y = rk[y];
		if (x > y) swap(x, y);
		++x;
		int z = __lg(y - x + 1);
		return min(st[z][x], st[z][y - (1 << z) + 1]);
	}
	SA(vector<int> a) :n(a.size()), st(__lg(n) + 1, vector<int>(n + 1)), sa(n), h(n)
	{
		int i, j, m, cnt;
		m = *min_element(all(a));
		for (int &x : a) x -= m;
		m = *max_element(all(a)) + 1;
		vector<int> id(n), s(max(n, m));
		rk = a;
		for (int i : rk) ++s[i];
		partial_sum(all(s), s.begin());
		for (i = n - 1; i >= 0; i--) sa[--s[rk[i]]] = i;
		for (j = 1; j <= n; j <<= 1)
		{
			fill_n(s.begin(), m, 0);
			cnt = j;
			iota(all(id) - (n - j), n - j);
			for (int i : sa) if (i >= j) id[cnt++] = i - j;
			for (int i : rk) ++s[i];
			partial_sum(all(s), s.begin());
			for (i = n - 1; i >= 0; i--) sa[--s[rk[id[i]]]] = id[i];
			id[sa[0]] = cnt = 0;
			for (i = 1; i < n; i++)
				if (sa[i] + j < n && sa[i - 1] + j < n && rk[sa[i]] == rk[sa[i - 1]] && rk[sa[i] + j] == rk[sa[i - 1] + j])
					id[sa[i]] = cnt;
				else
					id[sa[i]] = ++cnt;
			swap(rk, id);
			if ((m = cnt + 1) == n) break;
		}
		j = 0;
		for (i = 0; i < n; i++) if (rk[i])
		{
			cnt = sa[rk[i] - 1];
			while (i + j < n && cnt + j < n && a[i + j] == a[cnt + j]) ++j;
			h[rk[i]] = j;
			if (j) --j;
		}
		st[0] = h;
		for (j = 0; j < __lg(n); j++)
			for (i = 0, m = n - (1 << j + 1); i <= m; i++)
				st[j + 1][i] = min(st[j][i], st[j][i + (1 << j)]);
	}
	template<class T> SA(const T &s) :SA([&]() {
		vector<int> a; a.reserve(s.size());
		for (auto x : s) a.push_back(x);
		return a;
	}()) { }
};
