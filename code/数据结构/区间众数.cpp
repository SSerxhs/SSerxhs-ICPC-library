template<class T> struct mode//[0,n)
{
	int n, ksz, m;
	vector<T> b;
	vector<vector<int>> pos, f;
	vector<int> a, blk, id, l;
	mode(const vector<T> &c) :n(c.size()), ksz(max<int>(1, sqrt(n))), m((n + ksz - 1) / ksz), b(c),
		pos(n), f(m, vector<int>(m)), a(n), blk(n), id(n), l(m + 1)
	{
		int i, j, k;
		sort(all(b)); b.resize(unique(all(b)) - b.begin());
		for (i = 0; i < n; i++)
		{
			a[i] = lower_bound(all(b), c[i]) - b.begin();
			id[i] = pos[a[i]].size();
			pos[a[i]].push_back(i);
		}
		for (i = 0; i < n; i++) blk[i] = i / ksz;
		for (i = 0; i <= m; i++) l[i] = min(i * ksz, n);
		vector<int> cnt(b.size());
		for (i = 0; i < m; i++)
		{
			fill(all(cnt), 0);
			pair<int, int> cur = {0, 0};
			for (j = i; j < m; j++)
			{
				for (k = l[j]; k < l[j + 1]; k++) cmax(cur, pair{++cnt[a[k]], a[k]});
				f[i][j] = cur.second;
			}
		}
	}
	pair<T, int> ask(int L, int R)//返回最大众数
	{
		assert(0 <= L && L < R && R <= n);
		int val = blk[L] == blk[R - 1] ? 0 : f[blk[L] + 1][blk[R - 1] - 1], i;
		int cnt = lower_bound(all(pos[val]), R) - lower_bound(all(pos[val]), L);
		for (i = min(R, l[blk[L] + 1]) - 1; i >= L; i--)
		{
			auto &v = pos[a[i]];
			while (id[i] + cnt < v.size() && v[id[i] + cnt] < R) ++cnt, val = a[i];
			if (a[i] > val && id[i] + cnt - 1 < v.size() && v[id[i] + cnt - 1] < R) val = a[i];
		}
		for (i = max(L, l[blk[R - 1]]); i < R; i++)
		{
			auto &v = pos[a[i]];
			while (id[i] >= cnt && v[id[i] - cnt] >= L) ++cnt, val = a[i];
			if (a[i] > val && id[i] >= cnt - 1 && v[id[i] - cnt + 1] >= L) val = a[i];
		}
		return {b[val], cnt};
	}
};
