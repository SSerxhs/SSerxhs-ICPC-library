template<typename T> struct tork
{
	vector<T> a;
	int n;
	tork(const vector<T> &b):a(all(b))
	{
		sort(all(a));
		a.resize(unique(all(a))-a.begin());
		n=a.size();
	}
	tork(const T* first,const T* last):a(first,last)
	{
		sort(all(a));
		a.resize(unique(all(a))-a.begin());
		n=a.size();
	}
	void get(T &x) {x=lower_bound(all(a),x)-a.begin();}
	T operator[](const int &x) {return a[x];}
};