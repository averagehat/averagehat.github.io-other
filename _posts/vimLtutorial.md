Title: Vim Script 101 
Slug: 999 
Date:  January 2015
Category: Projects
Tags:
Author: Mike Panciera
Summary:



I'm currently working on [a plugin for vim](https://github.com/averagehat/phonim-jython). Along the way I've discovered a lot of intersting and useful productivity tricks. 

###Defining functions
```vim
   fun! Foo()
   " function body
   endfun

   function Foo()
   " function body
   endfunction
```

Create functions within the command window by putting each statement on a new line. i.e.
```vim
   :fun! Foo() [<CR>]
   :  second line
   : . . . . . 
```

Execute them from the commandline window:
```vim
   :call Foo()
```
Variables have several possible scopes:
`l` (local to function), `s` (local to individual scripts/files), `g` (global), `a` (a function parameter). 

`v` is a global like `g`, but has been predefined by vim. These cannot be defined by the user.

`b` (local to current buffer), `w` (local to the current winow) and `t` (local to current tab)  are less often used.

View defined variables:
```vim
   :let g:  "or v:, b:, etcl
```
By the way, we can use `"` for comments

View register contents:
```vim
   :reg
```

Variables are defined using let, e.g.:
```vim
 let l:bar = "red"
```
Variables defined or accessed within a function are by default in the local scope. Prepending with `g:` allows access to global variables.


Arrays are much like python lists. They are type-agnostic and can be sliced.
```vim
   :let l:a = [1, 2, 'foo', 'bar']
   :echo a[2:-1]` "prints ['foo', 'bar']
```
Strings can also be sliced withe same syntax.

String concatenation is php style:
```vim
let g:pair = 'eight' . 'eight'
```
Directly access the command line:
```vim
   :!echo -e "bash\nis\nfun" | cat > bar.txt 
```
Read files into the buffer:
```vim
   :r bar.txt
```
Produces:
```
bash
is
fun
```
Under your cursor.

Read command outputs into the buffer:
```vim
   :r !curl -s icanhazip.com
```

If your distribution of vim is compiled with python included (and it usually is) you can also access a python interpreter during your editing session.
```python
   :py print "ni!"
```
You can access vim from the internal python interpreter.
```python
   :py import vim
   :py vim.current.buffer[0] = 'Just answer the five (three!) questions'
``` 
You can go crazy with it if you like . . . 
```python
   :py import requests, lxml.etree
   :root = lxml.etree.HTML(requests.get('https://twitter.com/montypython').text)
   :redir @i| exe ':silent py for t in  root.xpath("//p[contains(@class, \"ProfileTweet-text\")]/text()"): print t.encode("utf8")' | redir END | $put i
```
Here, `:redir` redirects output of the commands between `:redir` and `:redir END` (which would usually be printed in the command window) to some register. Vim has 26+ register (all alphabetical letters and additional reserved registers). `@i` accesses the value at register `i`, telling vim to pipe output there. `|` stands in place of `<CR>` to separate commands. `exe` executes the given command as if it were in the command. We can use it when we need to make a command out of concatenating strings. In this case, it lets us wrap the python code in '' so that python does not try to interpret the next `|`. `silent` tells vim not to echo the result of the command in the command window. `redir END` will populate our register `i` -- it won't get updated until the redirect is finished. Finally, we execute `put i`, which inserts the contents of register `i` after the cursor's current line. The `$` (which generally indecates "last") tells put to instead append to the last line of the buffer.


There are some useful global variables
```vim
v:statusmsg    "usually holds the last item shown in the command window      
v:warningmsg   e.g. W10: Warning: Changing a readonly file
v:errmsg       e.g. E488: Trailing characters: put v:errmsg
```

Useful Motions (normal mode)
```vim
command    goes to
`f<char>` next instance of <char>
`)`       end of setnece
`%`       matching parens/brace
`$`       end of line
`G`       end of file
`}`       next paragraph
```
Additionally, we have `visual` mode. A quick preview of that:  
```vim
<C-v>  "To select columns. Any change propagates to all columns.
   :vs [filename] "split vertical window
   :split [filename] "split horizontally
```
`<C-w>` Is the window command suffix, so `<C-w> h` moves to the left widnow, `<C-w> l` to the right, `<C-w> k` moves to the upper window and `<C-w> j` moves to the lower one. I have the following mapping which forces this to work straight out of insert mode:
```vim
   :inoremap <C-w> <Esc> <C-w>
```
`<Esc>` exits to nromal mode. So does `<C-[>`; we can map the tab key to do it as well:
```vim
   :imap <tab> <Esc>
```
So what can we do with all this? Well, we have lots of useful variables, lots of registers, and marks, and even [ communication protocol for controlling external processes] (http://vimdoc.sourceforge.net/htmldoc/netbeans.html). My next post will be about the project that most of this study has been devoted to. For now, I'll go over a few of my personal settings and one trick that is a bit obscure but extremely powerful.
```vim
   :noremap <C-d> :call search('def\s\s*(' . expand('<cword>')) <CR>
```

The `search` function takes a string (remember that `.` performs string concatenation), searches the buffer for the regular expression, and moves the cursor to the the line of the found string. `search` returns the line of the found match, or `0` if no match was found (vim counts its buffer line from `1`). `<cword>` is an "Ex special character" (try `:h cmdline-special`). The `expand` function returns the special meaning of `<cword>`, which happens to be the current word under the cursor. So if the word under my cursor is "print_lines", search will be called on `def\s+print_line(`. So this obscure little function will take you to the function definiton of whatever function you're currently hovering over. IF it's in the file. But what if it's not? Let's try a function. 

```vim
fun! DefSearch()
  let l:regex = 'def\s\s*(' . expand('<cword>')
  if search(regex) == 0   " search failed
    silent 'grep -E "'  . regex . '" *.py'
  endif
endfun
```    
VimL's functions execute their code as if from the command-line (in vim, as if you prepended them with `:`). So `silent` `executes` the following command without echoing in the command window. It then execute's vim's built-in grep (which is a wrapper around bash's grep). grep's `E` allows you to use quote-wrapped regular expressions; -r forces a recursive search, if you like it. `*.py` will match any python files. Let's map it!
Food()
```vim
   :noremap <C-d> :call DefSearch() <CR>
```
But it's only useful if we're editing or reading a python file . . . we probably don't want to use it otherwise. So in the `.vimrc` file, we add:
```vim
au BufREadPost *.py :noremap <C-d> :call DefSearch() <CR>
```
Testing all this can be a bit of a drag, but as we are editing a vim script (or our .vimrc file) we can immmediately run it (and gain access to its globally-declared variables and functions) by calling `:source`. We can take advantage of anther "cmd-line-special" `%`, which in the command-line has the value of the current buffer name.
```vim
   :source %
```
Okay, one last trick. This one is real nifty. Info on it is in vim's extensive in-house documentation--just try `:h g@`. This trick allows you to capture user motions and reference them within a function. We do this by setting an 'operatorfunc' option. The result is a custom operator. Given a function Foo:

```vim
   :nmap <silent> <F4> :set opfunc=Echo<CR>g@
   :vmap <silent> <F4> :<C-U>call Echo(visualmode(), 1)<CR>
```
Note: `:vmap` is the equivalent of `:nmap` for visual mode. The function defition:

```vim
" uses [/] marks along with visual mode to yank a custom selection of text.
fun! Echo(type, ...)
  "backticks=' (goto mark), have to avoid out-quoting string
  "clear the @q register for use in this function.
  "@q is the reserved phonim register.
  let @q=""
  " see :h g@ for more info and how to save and restore a register, which
  " would allow us to use 'q' only temporarily and then restore it 
  if a:0  " Invoked from Visual mode, use '< and '> marks.
    silent exe "normal! `<" . a:type . '`>"qy'
  elseif a:type == 'line'
    silent exe 'normal! `[V`]"qy'
  elseif a:type == 'block' " column ('block') selection. 
    silent exe 'normal! `[\<C-V>`]"qy'
  "v -> visual mode but stay in-line
  else " Stay in-line
    silent exe 'normal! `[v`]"qy'  
  endif
  echo @q
endfun
```

This function echos into the command-window any text that the user selects. It saves the user's selection in the `q` register (could be any letter). It is possible to store and restore this register so as not to lose its original contents.  Now `:normal` executes normal-mode commands, and will use any mappings that we defined by `nmap`. `:normal!` overrides this use of mapping.  The `opfunc` works by setting `[` and `]` marks to the beginning and end of the user motiton (i.e. to the next whitespace if it is `W` or the next like if it is `j`). The `Echo` function simply yanks the text between these two marks and stores them in the `q` register. We can then do whatever we like with the text, in this case echo it. The if statements gaurantee that the selection will work the same in visual and normal mode (`v` is for inline visual selection; `V` is for column or "block" selection). 

In my next post I'll talk about an interesting application of this last technique--and VimL in general, and the reasoning behind all this research.
