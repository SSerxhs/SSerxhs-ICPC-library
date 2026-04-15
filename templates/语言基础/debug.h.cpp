template <class T, size_t size = tuple_size<T>::value> string to_dbg(T, string s = "") requires(not ranges::range<T>);
string to_dbg(auto x) requires requires(ostream &os) { os << x; }
{
	ostringstream os;
	os << x;
	return os.str();
}
string to_dbg(ranges::range auto x, string s = "") requires(not is_same_v<decltype(x), string>)
{
	for (auto xi : x) s += ", " + to_dbg(xi);
	return "[" + s.substr(2 * !!s.size()) + "]";
}
template <class T, size_t size> string to_dbg(T x, string s) requires(not ranges::range<T>)
{
	[&] <size_t... I>(index_sequence<I...>) { ((s += ", " + to_dbg(get<I>(x))), ...); } (make_index_sequence<size>());
	return "(" + s.substr(2 * !!s.size()) + ")";
}
#define dbg(...) cerr << __LINE__ << ": (" #__VA_ARGS__ ") = " << to_dbg(tuple(__VA_ARGS__)) << "\n"

