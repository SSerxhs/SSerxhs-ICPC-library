template<int M> struct sam//M：字符集大小
{
	vector<array<int, M>> c;
	vector<int> len, fa, ep;
	int np, cd;
	sam() :c(2), len(2), fa(2), ep(2), np(1), cd(0) { }
	void insert(int ch)
	{
		int p = np, q, nq;
		np = c.size();
		len.push_back(++cd);
		fa.push_back(0);
		c.push_back({ });
		ep.push_back(cd);
		while (p && !c[p][ch]) c[p][ch] = np, p = fa[p];
		if (!p)
		{
			fa[np] = 1;
			return;
		}
		q = c[p][ch];
		if (len[q] == len[p] + 1)
		{
			fa[np] = q;
			return;
		}
		nq = c.size();
		len.push_back(len[p] + 1);
		c.push_back(c[q]);
		fa.push_back(fa[q]);
		ep.push_back(ep[q]);
		fa[np] = fa[q] = nq;
		c[p][ch] = nq;
		while (c[p = fa[p]][ch] == q) c[p][ch] = nq;
	}
	vector<int> match(const string &s)//返回每个前缀最长匹配长度
	{
		vector<int> r;
		r.reserve(s.size());
		int p = 1, nl = 0;
		for (auto ch : s)
		{
			if (c[p][ch]) ++nl, p = c[p][ch];
			else
			{
				while (p && c[p][ch] == 0) p = fa[p];
				if (p == 0) p = 1, nl = 0; else nl = len[p] + 1, p = c[p][ch];
			}
			r.push_back(nl);
		}
		return r;
	}
	array<int, 3> max_match(const string &s)//返回长度，结尾（开）
	{
		array<int, 3> r{0, 0, 0};
		int p = 1, nl = 0, i = 0;
		for (auto ch : s)
		{
			if (c[p][ch]) ++nl, p = c[p][ch];
			else
			{
				while (p && c[p][ch] == 0) p = fa[p];
				if (p == 0) p = 1, nl = 0; else nl = len[p] + 1, p = c[p][ch];
			}
			cmax(r, array{nl, ep[p], i + 1});
			++i;
		}
		if (r[0] == 0) return { };
		return r;
	}
};


