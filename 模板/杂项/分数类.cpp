typedef long long ll;
struct frac
{
	ll x,y;
	frac(const ll &_t):x(_t),y(1) {}
	frac(const ll &_x,const ll &_y):x(_x),y(_y)
	{
		ll d=gcd(x,y);
		x/=d; y/=d;
		if (x<0||x==0&&y<0) x=-x,y=-y;
		assert(y);
	}
	frac operator+(const frac &o) const
	{
		ll d=lcm(y,o.y);
		return {d/y*x+d/o.y*o.x,d};
	}
	frac operator-(const frac &o) const
	{
		ll d=lcm(y,o.y);
		return {d/y*x-d/o.y*o.x,d};
	}
	frac operator*(const frac &o) const
	{
		return {x*o.x,y*o.y};
	}
	frac operator/(const frac &o) const
	{
		return {x*o.y,y*o.x};
	}
	frac &operator+=(const frac &o) { return *this=*this+o; }
	frac &operator-=(const frac &o) { return *this=*this-o; }
	frac &operator*=(const frac &o) { return *this=*this*o; }
	frac &operator/=(const frac &o) { return *this=*this/o; }
	bool operator<(const frac &o) const { return x*o.y<y *o.x; }
	bool operator==(const frac &o) const { return x*o.y==y*o.x; }
	bool operator>(const frac &o) const { return x*o.y>y*o.x; }
	bool operator!=(const frac &o) const { return x*o.y!=y*o.x; }
};
ostream &operator<<(ostream &cout,const frac &o)
{
	return cout<<o.x<<"/"<<o.y;
}