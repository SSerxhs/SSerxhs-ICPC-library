template<class T> struct dq
{
	vector<T> l, sl, r, sr;
	void push_front(const T &o)
	{
		sl.push_back(sl.size() ? o + sl.back() : o);
		l.push_back(o);
	}
	void push_back(const T &o)
	{
		sr.push_back(sr.size() ? sr.back() + o : o);
		r.push_back(o);
	}
	void pop_front()
	{
		if (l.size()) sl.pop_back(), l.pop_back();
		else
		{
			assert(r.size());
			int n = r.size(), m, i;
			if (m = n - 1 >> 1)
			{
				l.resize(m); sl.resize(m);
				for (i = 1; i <= m; i++) l[m - i] = r[i];
				sl[0] = l[0];
				for (i = 1; i < m; i++) sl[i] = l[i] + sl[i - 1];
			}
			for (i = m + 1; i < n; i++) r[i - (m + 1)] = r[i];
			m = n - (m + 1);
			r.resize(m); sr.resize(m);
			if (m)
			{
				sr[0] = r[0];
				for (i = 1; i < m; i++) sr[i] = sr[i - 1] + r[i];
			}
		}
	}
	void pop_back()
	{
		if (r.size()) sr.pop_back(), r.pop_back();
		else
		{
			assert(l.size());
			int n = l.size(), m, i;
			if (m = n - 1 >> 1)
			{
				r.resize(m); sr.resize(m);
				for (i = 1; i <= m; i++) r[m - i] = l[i];
				sr[0] = r[0];
				for (i = 1; i < m; i++) sr[i] = sr[i - 1] + r[i];
			}
			for (i = m + 1; i < n; i++) l[i - (m + 1)] = l[i];
			m = n - (m + 1);
			l.resize(m); sl.resize(m);
			if (m)
			{
				sl[0] = l[0];
				for (i = 1; i < m; i++) sl[i] = l[i] + sl[i - 1];
			}
		}
	}
	template<class TT> TT ask(TT r)
	{
		if (sl.size()) r = r + sl.back();
		if (sr.size()) r = r + sr.back();
		return r;
	}
	T ask()
	{
		assert(sl.size() || sr.size());
		if (sl.size() && sr.size()) return sl.back() + sr.back();
		return sl.size() ? sl.back() : sr.back();
	}
};

