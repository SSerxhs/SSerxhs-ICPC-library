template<class T>string to_dbg(T x)
{
	enum { r = ranges::range<T>, p = requires{tuple_size<T>(); } };
	if constexpr (requires{cout << x; } && (!r || requires{string(x); }))
	{
		ostringstream o;
		o << x;
		return o.str();
	}
	else
	{
		string s;
		if constexpr (r)
			for (auto y : x) s += ", " + to_dbg(y);
		else
			if constexpr (p)
				apply([&](auto...y) {((s += ", " + to_dbg(y)), ...); }, x);
		return (r ? "[" : "(") + s.substr(s[0] ? 2 : 0) + (r ? "]" : ")");
	}
}
#define dbg(...) cerr << __LINE__ << ": ("#__VA_ARGS__") = " << to_dbg(tuple(__VA_ARGS__)) << "\n"
