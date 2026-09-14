# CodeFact samples: 20 random Qwen3-valid items per category (2026-09-14 05:31 UTC)

For each item: the last lines of the prefix (`⟂` marks the split point), the true and foil tokens as the Qwen3 tokenizer sees them, and the subject token(s) that receive the noise. Check: is the subject the determinant of the answer?


## S1 (closing bracket)

**1. S1-00161** (), csn) true `)` foil `]` subject `(field` ("(" at char 18) backoff 0

```python
def any_slug_field(field, **kwargs⟂
```

**2. S1-00719** (}, csn) true `}` foil `]` subject `Ġ{` ("{" at char 84) backoff 0

```python
def update_setting(self, name, value):

    if value is not None:
        updates = {name : value⟂
```

**3. S1-01082** (), csn) true `)` foil `]` subject `('` ("(" at char 67) backoff 0

```python
def get_reserved_ip_address(self, name):
        _validate_not_none('name', name⟂
```

**4. S1-01027** (), csn) true `)` foil `]` subject `(location` ("(" at char 88) backoff 0

```python
def get_mnist(data_type="train", location="/tmp/mnist"):
    X, Y = mnist.read_data_sets(location, data_type⟂
```

**5. S1-00981** (), csn) true `)` foil `]` subject `(collection` ("(" at char 20) backoff 0

```python
def create_object_id(collection, vault, name, version⟂
```

**6. S1-00075** (), csn) true `Ġ)` foil `Ġ]` subject `(Ċ` ("(" at char 109) backoff 1

```python
        self.link = SecretLink.create(
            title,
            self.receiver,
            extra_data=dict(recid=self.recid),
            description=description,
            expires_at=expires_at,
        ⟂
```

**7. S1-00314** (), csn) true `)` foil `]` subject `(var` ("(" at char 19) backoff 0

```python
def parse_genotypes(variant, individuals, individual_positions⟂
```

**8. S1-00141** (}, csn) true `Ġ}` foil `Ġ]` subject `Ġ{Ċ` ("{" at char 47) backoff 1

```python
            'h': (-1, 0),
            'j': (0, 1),
            'k': (0, -1),
            'l': (1, 0),
            'y': (-1, -1),
            'u': (1, -1),
            'n': (1, 1),
            'b': (-1, 1),
        ⟂
```

**9. S1-00626** (], csn) true `]` foil `)` subject `[t` ("[" at char 266) backoff 0

```python
        slope = (Q[dtavgii] - Q[tavgii]*Q[davgii]/Q[sii]) \
                /(Q[tsqii] - Q[tavgii]**2/Q[sii])
        only_intercept=False
    else:
        only_intercept=True

    intercept = (Q[davgii] - Q[tavgii⟂
```

**10. S1-00977** (), mbpp) true `)` foil `]` subject `(nums` ("(" at char 12) backoff 0

```python
def sub_list(nums1,nums2⟂
```

**11. S1-00565** (), csn) true `)` foil `]` subject `(self` ("(" at char 15) backoff 0

```python
def get_key_for(self, address⟂
```

**12. S1-00597** (), mbpp) true `)` foil `]` subject `Ġ(` ("(" at char 51) backoff 0

```python
def centered_hexagonal_number(n):
  return 3 * n * (n - 1⟂
```

**13. S1-00827** (), csn) true `)` foil `]` subject `(queue` ("(" at char 258) backoff 0

```python
        _validate_not_none('queue_name', queue_name)
        request = HTTPRequest()
        request.method = 'PUT'
        request.host = self._get_host()
        request.path = '/' + _str(queue_name⟂
```

**14. S1-00477** (), csn) true `})` foil `}]` subject `({'` ("(" at char 127) backoff 1

```python
def hpo_term(self, hpo_id):
        LOG.debug("Fetching hpo term %s", hpo_id)

        return self.hpo_term_collection.find_one({'_id': hpo_id}⟂
```

**15. S1-01013** (], mbpp) true `]` foil `)` subject `[-` ("[" at char 41) backoff 0

```python
def rear_extract(test_list):
  res = [lis[-1⟂
```

**16. S1-00257** (), csn) true `)` foil `]` subject `(parser` ("(" at char 236) backoff 0

```python
    parser = create_parser()

    # Parse given arguments
    args = parser.parse_args()

    # Checking arguments
    check_arguments(args, parser)

    # BUSINESS LOGIC IS FOLLOWING
    run(parser, args⟂
```

**17. S1-00114** (), csn) true `)` foil `]` subject `(self` ("(" at char 14) backoff 0

```python
def add_ignore(self, *frames_or_fns⟂
```

**18. S1-00617** (), csn) true `)` foil `]` subject `(room` ("(" at char 78) backoff 0

```python
def update(self, roomId, title=None, **request_parameters):
        check_type(roomId, basestring, may_be_none=False⟂
```

**19. S1-00032** (], csn) true `']` foil `')` subject `['` ("[" at char 142) backoff 1

```python
def get_single_axis_values(self, axis, dataset):
		data_index = getattr(self, '%s_data_index' % axis)
		return [p[data_index] for p in dataset['data'⟂
```

**20. S1-01136** (], csn) true `]` foil `)` subject `[offset` ("[" at char 508) backoff 0

```python
        offset += int32.size

        if msg_id == MSG_EOF:
            break

        msg_size, = int32.unpack_from(data, offset)
        offset += int32.size

        msg_data = data[offset:offset + msg_size⟂
```


## S2 (block keyword)

**1. S2-01998** (elif_after_if, csn) true `Ġelif` foil `Ġelse` subject `Ġif` ("if" at char 135) backoff 1

```python
    yaml = ruamel.yaml.YAML()
    ruamel_data = yaml.load(file_text)

    if file_type in ('tasks', 'handlers'):
        ruamel_tasks = ruamel_data
        pyyaml_tasks = pyyaml_data
    ⟂
```

**2. S2-02086** (else_after_if, csn) true `Ġelse` foil `Ġelif` subject `Ġif` ("if" at char 84) backoff 1

```python


        def attempt_to_send(_):
            if peer not in self._connections:
                d = self._connect(peer)
                d.addCallback(attempt_to_send)
                return d
            ⟂
```

**3. S2-01204** (elif_after_if, csn) true `Ġelif` foil `Ġelse` subject `Ġif` ("if" at char 153) backoff 1

```python
        params = {'tag': tag} if tag else {}
        resp = self._fetch('screensaver', **params)

        if resp['data'] and resp['data']['id']:
            return self.gif(resp['data']['id'])
        ⟂
```

**4. S2-02112** (elif_after_if, csn) true `Ġelif` foil `Ġelse` subject `Ġif` ("if" at char 282) backoff 1

```python
    if data.dtype not in (np.uint8, np.int8):
        raise TypeError("unpack: dtype must be 8-bit")
    if nbit == 8:
        return data
    elif nbit == 4:
        data = unpack_4to8(data)
        return data
    ⟂
```

**5. S2-01745** (else_after_if, csn) true `Ġelse` foil `Ġelif` subject `Ġif` ("if" at char 154) backoff 1

```python
        if is_readable:
            self.cmd_queue.append('source ' + expanded_cmdfile)
        elif is_readable is None:
            self.errmsg("source file '%s' doesn't exist" % expanded_cmdfile)
        ⟂
```

**6. S2-01668** (elif_after_if, csn) true `Ġelif` foil `Ġelse` subject `Ġif` ("if" at char 58) backoff 1

```python
def models(cls, api_version=DEFAULT_API_VERSION):
        if api_version == '2016-04-01':
            from .v2016_04_01 import models
            return models
        ⟂
```

**7. S2-01409** (elif_after_if, mbpp) true `Ġelif` foil `Ġelse` subject `Ġif` ("if" at char 243) backoff 1

```python
            c0 += 1;
        elif (s1[i] == '1' and s2[i] == '0') :
            c1 += 1;
    result = c0 // 2 + c1 // 2;
    if (c0 % 2 == 0 and c1 % 2 == 0) :
        return result;
    ⟂
```

**8. S2-01850** (except_after_try, csn) true `Ġexcept` foil `Ġfinally` subject `Ġtry` ("try" at char 37) backoff 1

```python
        try:
            return (self.min_x <= other.max_x and
                    self.max_x >= other.min_x and
                    self.min_y <= other.max_y and
                    self.max_y >= other.min_y)
        ⟂
```

**9. S2-01262** (else_after_if, csn) true `Ġelse` foil `Ġelif` subject `Ġif` ("if" at char 319) backoff 1

```python
        case_id=case_id,
    )

    if case_obj is None:
        raise DataNotFoundError("no case found")

    if not case_obj.get('delivery_report'):
        _put_report_in_case_root(case_obj, report_path)
    ⟂
```

**10. S2-01245** (else_after_if, csn) true `Ġelse` foil `Ġelif` subject `Ġif` ("if" at char 508) backoff 1

```python
                            data=json.dumps(body), verify=self.verify)
        if resp.status_code == 201:
            self.logger.debug("Logical interface rule created")
        ⟂
```

**11. S2-01252** (else_after_if, csn) true `Ġelse` foil `Ġelif` subject `Ġif` ("if" at char 353) backoff 1

```python
                if not user:
                    logger.warning("Missing user info for %s", comment['url'])
                    comment['user_data'] = None
                ⟂
```

**12. S2-02308** (elif_after_if, csn) true `Ġelif` foil `Ġelse` subject `Ġif` ("if" at char 181) backoff 1

```python
                print("-- sys ----------------------------------------")
                for line in info_formatter(self.coverage.sysinfo()):
                    print(" %s" % line)
            ⟂
```

**13. S2-01218** (elif_after_if, csn) true `Ġelif` foil `Ġelse` subject `Ġif` ("if" at char 60) backoff 1

```python
def p_colspec(p):
    rest = p[3] if len(p) > 3 else []
    if p[1] == "*":
        p[0] = [{"type": "star"}]
    elif isinstance(p[1], dict) and p[1].get("type") == "function":
        p[0] = [p[1], *rest]
    ⟂
```

**14. S2-01980** (else_after_if, csn) true `Ġelse` foil `Ġelif` subject `Ġif` ("if" at char 279) backoff 1

```python
    if len(later_prefix) == 1:
        later_prefix = ' '*len(prefix)
    line = line + '\n'.join([later_prefix + x for x in textblock[1:]])
    if line[-1] != '\n':
        return line + '\n'
    ⟂
```

**15. S2-01643** (elif_after_if, csn) true `Ġelif` foil `Ġelse` subject `Ġif` ("if" at char 131) backoff 1

```python
def has_leading_dir(paths):
    common_prefix = None
    for path in paths:
        prefix, rest = split_leading_dir(path)
        if not prefix:
            return False
        ⟂
```

**16. S2-02064** (else_after_if, csn) true `Ġelse` foil `Ġelif` subject `Ġif` ("if" at char 37) backoff 1

```python
def _log_message(self, msg):
        if is_type(self._logging_dest, str):
            with open(self._logging_dest, "at", encoding="utf-8") as f:
                f.write(msg)
        ⟂
```

**17. S2-01259** (except_after_try, csn) true `Ġexcept` foil `Ġfinally` subject `Ġtry` ("try" at char 50) backoff 1

```python
        try:
            values = set(node.infer())
            current = set(parent.instance_attrs_type[node.attrname])
            parent.instance_attrs_type[node.attrname] = list(current | values)
        ⟂
```

**18. S2-02280** (except_after_try, csn) true `Ġexcept` foil `Ġfinally` subject `Ġtry` ("try" at char 42) backoff 1

```python
def _convert_to_float_if_possible(s):
    try:
        ret = float(s)
    ⟂
```

**19. S2-01654** (elif_after_if, csn) true `Ġelif` foil `Ġelse` subject `Ġif` ("if" at char 244) backoff 1

```python
            except AttributeError:
                pass
            else:
                ret = ensure_fromlist(mod, all, buf, 1)
                if not ret:
                    return 0
        ⟂
```

**20. S2-02096** (else_after_if, csn) true `Ġelse` foil `Ġelif` subject `Ġif` ("if" at char 50) backoff 1

```python
def to_lisp(o, keywordize_keys: bool = True):
    if not isinstance(o, (dict, frozenset, list, set, tuple)):
        return o
    ⟂
```


## S3 (keyword completion)

**1. S3-03419** (for_in, csn) true `Ġin` foil `,` subject `Ġfor` ("for" at char 102) backoff 0

```python
def export_directives(self):

        directives_json = {}

        # Skip first init process
        for p⟂
```

**2. S3-03537** (for_in, humaneval) true `Ġin` foil `,` subject `Ġfor` ("for" at char 152) backoff 0

```python
from typing import List


def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    if not numbers:
        return []

    result = []

    for n⟂
```

**3. S3-02877** (for_in, csn) true `Ġin` foil `,` subject `Ġfor` ("for" at char 26) backoff 0

```python
def _init(self):

        for attr⟂
```

**4. S3-03108** (if_colon, csn) true `:` foil `Ġand` subject `Ġif` ("if" at char 225) backoff 0

```python
        # Reconnect attempt at self.reconnect_interval
        self.log.debug("reconnect(): Initialzion reconnect sequence..")
        self.connected.clear()
        self.reconnect_required.set()
        if self.socket⟂
```

**5. S3-02872** (for_in, csn) true `Ġin` foil `,` subject `Ġfor` ("for" at char 235) backoff 0

```python
  if comment is not None:
    write_comment(fh, comment)

  if timestamp:
    write_comment(fh, time.strftime('%a %b %d %H:%M:%S %Z %Y'))

  if hasattr(props, 'keys'):
    for key⟂
```

**6. S3-02848** (for_in, csn) true `Ġin` foil `,` subject `Ġfor` ("for" at char 359) backoff 0

```python
    atags = root.cssselect('a')
    hrefs = [a.attrib['href'] for a in atags]
    # !!! This does the wrong thing for bbc.co.uk/index.html
    hrefs = [h if h.startswith('http') else '/'.join([url, h]) for h⟂
```

**7. S3-03345** (if_colon, csn) true `:` foil `Ġand` subject `Ġif` ("if" at char 466) backoff 0

```python
            for ann in tier.get_intervals(True):
                if tier.tier_type == 'TextTier':
                    ann = (ann[0], ann[0]+pointlength, ann[1])
                if ann[2].strip() or not skipempty⟂
```

**8. S3-02994** (if_colon, csn) true `:` foil `Ġand` subject `Ġif` ("if" at char 131) backoff 0

```python
def cut(self, breaks, labels=None, include_lowest=False, right=True, dig_lab=3):
        assert_is_type(breaks, [numeric])
        if self.ncols != 1⟂
```

**9. S3-02444** (if_colon, csn) true `:` foil `Ġand` subject `Ġif` ("if" at char 202) backoff 0

```python
def encodeLength(value):
    encoded = bytearray()
    while True:
        digit = value % 128
        value //= 128
        if value > 0:
            digit |= 128
        encoded.append(digit)
        if value <= 0⟂
```

**10. S3-03254** (for_in, csn) true `Ġin` foil `,` subject `Ġfor` ("for" at char 323) backoff 0

```python

    parse_args(argv)

    if (g_parse_log_path is None):
        print("")
        print("ERROR: -f not specified")
        usage()

    d = Dataset(g_parse_log_path)
    d.parse()

    d.emit_header()
    for i⟂
```

**11. S3-03544** (for_in, csn) true `Ġin` foil `,` subject `Ġfor` ("for" at char 206) backoff 0

```python
def packIntf(intf, masterDirEqTo=DIRECTION.OUT, exclude=None):
    if not intf._interfaces:
        if intf._masterDir == masterDirEqTo:
            return intf._sig
        return None

    res = None
    for i⟂
```

**12. S3-02604** (for_in, mbpp) true `Ġin` foil `,` subject `Ġfor` ("for" at char 67) backoff 0

```python
R = 3
C = 3
def min_cost(cost, m, n):
	tc = [[0 for x in range(C)] for x⟂
```

**13. S3-02780** (if_colon, csn) true `:` foil `Ġand` subject `Ġif` ("if" at char 86) backoff 0

```python
def set_validation(self, batch_size, X_val, Y_val, trigger, val_method=None):
        if val_method is None⟂
```

**14. S3-03008** (for_in, csn) true `Ġin` foil `,` subject `Ġfor` ("for" at char 285) backoff 0

```python

  ndims = tf.convert_to_tensor(value=ndims, name='ndims', dtype=tf.int32)
  ndims_ = tf.get_static_value(ndims)

  if _is_list_like(axis) and ndims_ is not None:
    # Static case
    positive_axis = []
    for a⟂
```

**15. S3-02647** (if_colon, csn) true `:` foil `Ġand` subject `Ġif` ("if" at char 82) backoff 0

```python
def _contextualise_connection(self, connection):

        ctx = stack.top
        if ctx is not None⟂
```

**16. S3-03082** (for_in, csn) true `Ġin` foil `,` subject `Ġfor` ("for" at char 210) backoff 0

```python
def get_config(app, prefix='hive_'):
    items = app.config.items()
    prefix = prefix.upper()

    def strip_prefix(tup):
        return (tup[0].replace(prefix, ''), tup[1])

    return dict([strip_prefix(i) for i⟂
```

**17. S3-03429** (for_in, csn) true `Ġin` foil `,` subject `Ġfor` ("for" at char 644) backoff 0

```python
        msg += "[%s] %s\n" % (time.strftime("%H:%M:%S"), endpoint)
        if params is not None: msg += "     params: {%s}\n" % ", ".join("%s:%s" % item for item⟂
```

**18. S3-03267** (for_in, mbpp) true `Ġin` foil `,` subject `Ġfor` ("for" at char 196) backoff 0

```python
from itertools import groupby
def group_element(test_list):
  res = dict()
  for key, val in groupby(sorted(test_list, key = lambda ele: ele[1]), key = lambda ele: ele[1]):
    res[key] = [ele[0] for ele⟂
```

**19. S3-03443** (if_colon, csn) true `:` foil `Ġand` subject `Ġif` ("if" at char 352) backoff 0

```python
        # times when `logging.shutdown` is called.
        if self.closed:
            return

        super().close()

        if not self.upload_on_close⟂
```

**20. S3-02788** (if_colon, csn) true `:` foil `Ġand` subject `Ġif` ("if" at char 26) backoff 0

```python
def tables(self):
        if self.table is None⟂
```


## R1 (variable recall)

**1. R1-04257** (store, csn) true `Ġfirst` foil `Ġshape` subject `Ġfirst` ("first" at char 46) backoff 1

```python
def shape(self):
        # TODO cache
        first = self.first().shape
        shape = self._rdd.map(lambda x: x.shape[0]).sum()
        return (shape,) + ⟂
```

**2. R1-04213** (store, mbpp) true `(temp` foil `(res` subject `Ġtemp` ("temp" at char 72) backoff 1

```python
from collections import defaultdict

def most_occurrences(test_list):
  temp = defaultdict(int)
  for sub in test_list:
    for wrd in sub.split():
      temp[wrd] += 1
  res = max(⟂
```

**3. R1-04684** (param, csn) true `Ġnode` foil `Ġbase` subject `Ġnode` ("node" at char 62) backoff 1

```python
    frame: astroid.node_classes.NodeNG, node: astroid.node_classes.NodeNG
) -> bool:
    try:
        bases = frame.bases
    except AttributeError:
        return False
    for base in bases:
        if ⟂
```

**4. R1-04696** (param, csn) true `Ġstring` foil `Ġcursor` subject `Ġstring` ("string" at char 28) backoff 1

```python
        cursor.removeSelectedText()

        # Insert new text with continuation prompts.
        self._insert_plain_text_into_buffer(self._get_prompt_cursor(), ⟂
```

**5. R1-04453** (store, csn) true `Ġpostgres` foil `Ġconn` subject `Ġpostgres` ("postgres" at char 35) backoff 1

```python
def _query_postgres(self):
        postgres = PostgresHook(postgres_conn_id=self.postgres_conn_id)
        conn = ⟂
```

**6. R1-03676** (store, csn) true `Ġselect` foil `Ġoutputs` subject `Ġselect` ("select" at char 89) backoff 1

```python
def _fill_enclosure(self, enclosure: Dict[RtlSignalBase, HdlStatement]) -> None:
        select = []
        outputs = self._outputs
        for e in enclosure.keys():
            if e in outputs:
                ⟂
```

**7. R1-04642** (param, csn) true `(group` foil `(ep` subject `(group` ("group" at char 20) backoff 1

```python
def get_group_named(group, path=None):
    result = {}
    for ep in get_group_all(⟂
```

**8. R1-04126** (store, mbpp) true `Ġmedian` foil `Ġheight` subject `Ġmedian` ("median" at char 43) backoff 1

```python
def median_trapezium(base1,base2,height):
 median = 0.5 * (base1+ base2)
 return ⟂
```

**9. R1-04477** (param, mbpp) true `Ġtext` foil `Ġpatterns` subject `(text` ("text" at char 25) backoff 1

```python
import re
def text_match(text):
  patterns = 'a.*?b$'
  if re.search(patterns,  ⟂
```

**10. R1-04499** (store, csn) true `Ġshape` foil `Ġtensor` subject `Ġshape` ("shape" at char 71) backoff 1

```python
def constant_value_as_shape(tensor):  # pylint: disable=invalid-name
  shape = tf.get_static_value(tensor)
  if ⟂
```

**11. R1-03977** (store, csn) true `Ġfuture` foil `Ġsegment` subject `Ġfuture` ("future" at char 112) backoff 1

```python
            future = self.executor.submit(self.fetch, segment,
                                          retries=self.retries)
        else:
            future = None

        self.queue(self.futures, (segment, ⟂
```

**12. R1-04397** (store, mbpp) true `(val` foil `(key` subject `Ġval` ("val" at char 58) backoff 1

```python
def assign_elements(test_list):
  res = dict()
  for key, val in test_list:
    res.setdefault(val, [])
    res.setdefault(key, []).append(⟂
```

**13. R1-04790** (param, mbpp) true `Ġnumbers` foil `Ġtotal` subject `(numbers` ("numbers" at char 12) backoff 1

```python
def sum_num(numbers):
    total = 0
    for x in ⟂
```

**14. R1-04414** (param, csn) true `Ġbegin` foil `Ġcells` subject `Ġbegin` ("begin" at char 36) backoff 1

```python
def _build_line(colwidths, padding, begin, fill, sep, end):
    cells = [fill * (w + 2 * padding) for w in colwidths]
    return _build_row(cells, 0, ⟂
```

**15. R1-03789** (store, csn) true `Ġshape` foil `Ġmat` subject `Ġshape` ("shape" at char 98) backoff 1

```python
def is_square_matrix(mat):
    mat = np.array(mat)
    if mat.ndim != 2:
        return False
    shape = mat.shape
    return ⟂
```

**16. R1-04551** (store, csn) true `(path` foil `(contents` subject `Ġpath` ("path" at char 70) backoff 1

```python
def __get_empty_config(self):
        self._generate_config()
        path = self._get_config_path()
        with open(path, 'r') as readable:
            contents = readable.read()
        os.remove(⟂
```

**17. R1-04704** (param, csn) true `Ġnew` foil `Ġedges` subject `Ġnew` ("new" at char 39) backoff 1

```python
def _on_edges(self, object, name, old, new):
        if name == "edges_items":
            edges = new.added
        elif name == "edges":
            edges = ⟂
```

**18. R1-03835** (store, mbpp) true `mid` foil `high` subject `Ġmid` ("mid" at char 121) backoff 0

```python
def find_Max(arr,low,high):
    if (high < low):
        return arr[0]
    if (high == low):
        return arr[low]
    mid = low + (high - low) // 2
    if (⟂
```

**19. R1-03958** (store, csn) true `(table` foil `(df` subject `Ġtable` ("table" at char 166) backoff 1

```python
        # get the table
        url = self._subpage_url('splits', year)
        doc = pq(sportsref.utils.get_html(url))
        table = doc('table#advanced_splits')
        df = sportsref.utils.parse_table(⟂
```

**20. R1-04729** (store, csn) true `Ġinterceptor` foil `Ġkwargs` subject `Ġinterceptor` ("interceptor" at char 122) backoff 1

```python
def interceptable(func):
  @functools.wraps(func)
  def func_wrapped(*args, **kwargs):
    with get_next_interceptor() as interceptor:
      return ⟂
```


## R2 (attribute / API recall)

**1. R2-05203** (local_list, csn) true `.append` foil `.extend` subject `Ġ[]Ċ` ("[]" at char 124) backoff 1

```python
        divisor = 1000 if decimal else 1024
        num = []
        c = ""
        for c in human_readable_str:
            if c not in cls.digits:
                break
            num.⟂
```

**2. R2-05180** (local_str, csn) true `.format` foil `.join` subject `Ġ" Amb iguous Ġselector Ġ'{} ', Ġmatches Ġ{} ."Ċ` (""Ambiguous selector '{}', matches {}."" at char 224) backoff 1

```python
    if not matching_selectors:
      return default
    if len(matching_selectors) > 1:
      err_str = "Ambiguous selector '{}', matches {}."
      raise KeyError(err_str.⟂
```

**3. R2-05302** (local_list, csn) true `.append` foil `.extend` subject `Ġ[]Ċ` ("[]" at char 70) backoff 1

```python
                # recursively call this function
                install_requires += parse_reqs(req_path=line[3:])

            else:
                # add the line as a new requirement
                install_requires.⟂
```

**4. R2-05551** (module_re, mbpp) true `.sub` foil `.match` subject `Ġre` ("re" at char 7) backoff 1

```python
import re
def replace_specialchar(text):
 return (re.⟂
```

**5. R2-04830** (local_list, csn) true `.extend` foil `.append` subject `Ġ[]Ċ` ("[]" at char 116) backoff 1

```python
        super(NoseExclude, self).options(parser, env)
        env_dirs = []
        if 'NOSE_EXCLUDE_DIRS' in env:
            exclude_dirs = env.get('NOSE_EXCLUDE_DIRS','')
            env_dirs.⟂
```

**6. R2-05281** (local_list, mbpp) true `.append` foil `.extend` subject `Ġ[]Ċ` ("[]" at char 30) backoff 1

```python
def get_key(dict):
    list = []
    for key in dict.keys():
        list.⟂
```

**7. R2-04844** (local_list, csn) true `.append` foil `.extend` subject `Ġ[]Ċ` ("[]" at char 127) backoff 1

```python
                if t1.is_polymorphic:
                    continue
                if (s.tret in self.mapTypeTranslate):
                    if (t2 in self.mapTypeTranslate[t1]):
                        collect.⟂
```

**8. R2-05115** (module_re, mbpp) true `.search` foil `.match` subject `Ġre` ("re" at char 7) backoff 1

```python
regex = '''^(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.(
			25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.(
			25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.(
			25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)$'''
def check_IP(Ip):
	if(re.⟂
```

**9. R2-05521** (local_set, mbpp) true `.add` foil `.remove` subject `Ġset ()Ċ` ("set()" at char 190) backoff 1

```python
        sum_fact2 = sum([fact for fact in range(1, sum_fact) if sum_fact % fact == 0])
        if num == sum_fact2 and num != sum_fact:
            amicables.⟂
```

**10. R2-05430** (local_list, csn) true `.append` foil `.extend` subject `Ġ[]Ċ` ("[]" at char 258) backoff 1

```python
                    body = fp.read()
                break
            elif os.path.isdir(os.path.join(filename,f)):
                f+='/'
            files.⟂
```

**11. R2-05408** (local_list, csn) true `.append` foil `.extend` subject `Ġ[]Ċ` ("[]" at char 567) backoff 1

```python
        nodes = []
        for node in self._multi_graph.nodes():
            if node.type == "op":
                if op is None or isinstance(node.op, op):
                    nodes.⟂
```

**12. R2-05393** (local_list, csn) true `.extend` foil `.append` subject `Ġ[]Ċ` ("[]" at char 112) backoff 1

```python
def _resolve_distribution_names(dist_fn_args, dist_names, leaf_name):
  if dist_names is None:
    dist_names = []
  else:
    dist_names = dist_names.copy()
  n = len(dist_fn_args)
  dist_names.⟂
```

**13. R2-05204** (local_list, humaneval) true `.append` foil `.extend` subject `Ġ[ 1 , Ġ 3 ]Ċ` ("[1, 3]" at char 60) backoff 1

```python

def tri(n):
    if n == 0:
        return [1]
    my_tri = [1, 3]
    for i in range(2, n + 1):
        if i % 2 == 0:
            my_tri.⟂
```

**14. R2-05463** (local_dict, csn) true `.items` foil `.get` subject `Ġ{" SYM {}". format (re .sub ("[ \ [\ ]\ - | =: ]", Ġ"_ ", Ġpc )): Ġpc Ċ ĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠ Ġfor Ġi , Ġpc Ġin Ġenumerate (per f count ers )}Ċ` ("{"SYM{}".format(re.sub("[\[\]\-|=:]", "_", pc)): pc
                         for i, pc in enumerate(perfcounters)}" at char 294) backoff 1

```python
        temp_metric = metric
        temp_pc_names = {"SYM{}".format(re.sub("[\[\]\-|=:]", "_", pc)): pc
                         for i, pc in enumerate(perfcounters)}
        for var_name, pc in temp_pc_names.⟂
```

**15. R2-04974** (local_list, csn) true `.append` foil `.extend` subject `Ġ[]Ċ` ("[]" at char 38) backoff 1

```python
def join_lines(iterator):
    lines = []
    for line in iterator:
        if not line.endswith('\\'):
            if lines:
                lines.⟂
```

**16. R2-04972** (local_list, csn) true `.extend` foil `.append` subject `Ġlist (it ertools .is lice (it , Ġlength ))Ċ` ("list(itertools.islice(it, length))" at char 97) backoff 1

```python
    while len(results) == length:
        yield results
        results = results[length-overlap:]
        results.extend(itertools.islice(it, length-overlap))
    if padding and results:
        results.⟂
```

**17. R2-05315** (local_list, csn) true `.sort` foil `.append` subject `Ġ[ symbol , Ġother _symbol ]Ċ` ("[symbol, other_symbol]" at char 76) backoff 1

```python
def _raise_duplicate_symbol(msgid, symbol, other_symbol):
        symbols = [symbol, other_symbol]
        symbols.⟂
```

**18. R2-05032** (local_dict, csn) true `.update` foil `.get` subject `Ġ{}Ċ` ("{}" at char 64) backoff 1

```python
def _read_params(input_file, json_body, p_params):
    params = {}
    try:
        if input_file:
            params.update(json.load(input_file))
        if json_body is not None:
            params.⟂
```

**19. R2-04812** (local_list, csn) true `.append` foil `.extend` subject `Ġ[]Ċ` ("[]" at char 107) backoff 1

```python
    for index in _get_front_idxs_from_id(onset_fronts, onset_front_id):
        offset_fidx, offset_sidx = _lookup_offset_by_onset_idx(index, onsets, offsets)
        corresponding_offsets.⟂
```

**20. R2-05590** (local_list, csn) true `.append` foil `.extend` subject `Ġ[]Ċ` ("[]" at char 243) backoff 1

```python
        lines = cursor.fetchall()
        if not lines:
            # table does not exist
            return True
        types = {}
        keys = []
        for line in lines:
            keys.⟂
```


## R3 (constant recall)

**1. R3-06017** (str, csn) true `service` foil `?` subject `service` ("service" at char 217) backoff 0

```python

    query_string = []
    if url.find('?') != -1:
        query_string = cgi.parse_qsl(url.split('?')[1])

    params = [x[0] for x in query_string]

    if 'service' not in params:
        query_string.append(('⟂
```

**2. R3-06735** (str, csn) true `image` foil `services` subject `image` ("image" at char 79) backoff 0

```python
def get_dusty_images():
    specs = get_specs()
    dusty_image_names = [spec['image'] for spec in specs['apps'].values() + specs['services'].values() if '⟂
```

**3. R3-06754** (str, csn) true `Ġ` foil `description` subject `Ġ'` (" " at char 149) backoff 0

```python
    if header.lower() == 'description':  # preserve newlines
        return '\n'.join([x[8:] if x.startswith(' ' * 8) else x
                          for x in txt.strip().splitlines()])
    else:
        return '⟂
```

**4. R3-06087** (str, csn) true `name` foil `values` subject `name` ("name" at char 102) backoff 0

```python
        if field["type"].startswith("enum"):
            ordered_schemas[field_schema_name] = field["values"]
        else:
            field_schema = schemas_map[field_schema_name]
            if field_schema["⟂
```

**5. R3-06449** (str, csn) true `cs` foil `sr` subject `cs` ("cs" at char 512) backoff 0

```python
                    thrift_annotation.host,
                )

        if 'cs' in all_annotations and 'sr' not in all_annotations:
            kind = Kind.CLIENT
            timestamp = all_annotations['⟂
```

**6. R3-06680** (int, csn) true `3` foil `2` subject `3` ("3" at char 69) backoff 0

```python
def minify_hex(_hex):
    size = len(_hex.strip('#'))
    if size == 3:
        return _hex
    elif size == 6:
        if _hex[1] == _hex[2] and _hex[⟂
```

**7. R3-06322** (str, csn) true `minor` foil `major` subject `minor` ("minor" at char 236) backoff 0

```python
    if match is None:
        raise ValueError('%s is not valid SemVer string' % version)

    verinfo = match.groupdict()

    verinfo['major'] = int(verinfo['major'])
    verinfo['minor'] = int(verinfo['⟂
```

**8. R3-06341** (str, csn) true `width` foil `device` subject `width` ("width" at char 102) backoff 0

```python
def save(url, *args, **kwargs):

    device = heimdallDevice(kwargs.get('device', None))

    kwargs['width'] = kwargs.get('⟂
```

**9. R3-06565** (str, csn) true `capture` foil `files` subject `capture` ("capture" at char 370) backoff 0

```python
    validator = Validator()
    val = cfg.validate(validator)
    if val is not True:
        raise ValueError('Invalid configuration: %s' % val)
    if len(cfg['capture']['files']) != len(cfg['⟂
```

**10. R3-06164** (str, csn) true `x` foil `y` subject `x` ("x" at char 191) backoff 0

```python
        for point in points:
            if point['x'] == start:
                include = True
            if include:
                data.append(point['y'])
            if end is not None and point['⟂
```

**11. R3-05607** (str, csn) true `import` foil `.` subject `import` ("import" at char 109) backoff 0

```python
    # Get the mappings and keys.
    mapping = { ".": "" }
    if (('import' in autooptions) and
        ('directory-mapping' in autooptions['import'])):
        mapping = autooptions['⟂
```

**12. R3-06406** (str, csn) true `lx` foil `l` subject `lx` ("lx" at char 96) backoff 0

```python
    edges = self.split("edges", at = "coords").unstack()
    edges["lx"] = edges.x[1]-edges.x[0]
    edges["ly"] = edges.y[1]-edges.y[0]
    edges["lz"] = edges.z[1]-edges.z[0]
    edges["l"] = np.linalg.norm(edges[["⟂
```

**13. R3-06677** (int, csn) true `3` foil `2` subject `3` ("3" at char 131) backoff 0

```python
        }

        bits += [0] * (3 - len(bits)) # pad to 3 digits

        # Increment the version
        bits[indexes[type]] += 1

        # Set the subsequent digits to 0
        for i in range(indexes[type] + 1, ⟂
```

**14. R3-05869** (str, csn) true `libs` foil `depends` subject `libs` ("libs" at char 128) backoff 0

```python
def _expand_libs_in_apps(specs):
    for app_name, app_spec in specs['apps'].iteritems():
        if 'depends' in app_spec and 'libs' in app_spec['depends']:
            app_spec['depends']['libs'] = _get_dependent('⟂
```

**15. R3-06690** (str, csn) true `name` foil `assign` subject `name` ("name" at char 140) backoff 0

```python
            category='case',
            verb='assign',
            subject=case['display_name']
        )
        LOG.info("Updating {0} to be assigned with {1}"
                    .format(case['display_name'], user['⟂
```

**16. R3-06781** (int, csn) true `6` foil `5` subject `6` ("6" at char 50) backoff 0

```python
def parse_unix_mode(s):
        parse_rw = {"rw": 6, "r-": 4, "-w": 2, "--": 0}
        mode = 0
        mode |= parse_rw[s[0:2]] << 6
        mode |= parse_rw[s[3:5]] << 3
        mode |= parse_rw[s[⟂
```

**17. R3-06029** (str, csn) true `type` foil `Bool` subject `type` ("type" at char 119) backoff 0

```python
            return self.getOptionAsString(name)
        elif PyOptionList[name]['type'] == "Bool":
            return self.getOptionAsBool(name)
        elif PyOptionList[name]['⟂
```

**18. R3-06494** (str, csn) true `path` foil `get` subject `path` ("path" at char 101) backoff 0

```python
def list_files(self, id=None, path="/"):
        if id:
            return self.request(id, params={"path": path}, method="get").json()
        else:
            return self.request(params={"⟂
```

**19. R3-05712** (str, csn) true `new` foil `mode` subject `new` ("new" at char 104) backoff 0

```python
def cdr(ol,**kwargs):
    if('mode' in kwargs):
        mode = kwargs['mode']
    else:
        mode = "new"
    if(mode == "⟂
```

**20. R3-06611** (int, csn) true `2` foil `5` subject `2` ("2" at char 327) backoff 0

```python
    if s.traffic:
        s._ctx.fill(
            s.traffic.r,
            s.traffic.g,
            s.traffic.b,
            s.traffic.a * alpha
        )
        s._ctx.oval(node.x-r, node.y-r, r*2, r*⟂
```
