var promptbook_click_image = function(){
    if (!this.classList.contains("transform")){        
        var gallery = promptbook_get_parent_by_class(this, "promptbook_cantainor");
        var buttons = gallery.querySelectorAll(".gallery-item");
        var i = 0;
        var hidden_list = [];
        buttons.forEach(function(e){
            if (e.style.display == "none"){
                hidden_list.push(i);
            }
            i += 1;
        })
        if (hidden_list.length > 0){
            setTimeout(promptbook_hide_buttons, 10, hidden_list, gallery);
        }        
    }    
    promptbook_set_image_info(this);
}

function promptbook_get_parent_by_class(item, class_name){
    var parent = item.parentElement;
    while(!parent.classList.contains(class_name)){
        parent = parent.parentElement;
    }
    return parent;  
}

function promptbook_get_parent_by_tagname(item, tagname){
    var parent = item.parentElement;
    tagname = tagname.toUpperCase()
    while(parent.tagName != tagname){
        parent = parent.parentElement;
    }  
    return parent;
}

function promptbook_hide_buttons(hidden_list, gallery){
    var buttons = gallery.querySelectorAll(".gallery-item");
    var num = 0;
    buttons.forEach(function(e){
        if (e.style.display == "none"){
            num += 1;
        }
    });
    if (num == hidden_list.length){
        setTimeout(promptbook_hide_buttons, 10, hidden_list, gallery);
    } 
    for( i in hidden_list){
        buttons[hidden_list[i]].style.display = "none";
    }    
}

function promptbook_set_image_info(button){
    var buttons = promptbook_get_parent_by_tagname(button, "DIV").querySelectorAll(".gallery-item");
    var index = -1;
    var i = 0;
    buttons.forEach(function(e){
        if(e == button){
            index = i;
        }
        if(e.style.display != "none"){
            i += 1;
        }        
    });
    var gallery = promptbook_get_parent_by_class(button, "promptbook_cantainor");
    var set_btn = gallery.querySelector(".promptbook_set_index");
    var curr_idx = set_btn.getAttribute("img_index", index);  
    if (curr_idx != index) {
        set_btn.setAttribute("img_index", index);        
    }
    set_btn.click();
    
}

function promptbook_get_current_img(tabname, img_index, page_index){
    return [
        tabname, 
        gradioApp().getElementById(tabname + '_promptbook_set_index').getAttribute("img_index"),
        page_index
    ];
}

function promptbook_delete(del_num, tabname, image_index){
    image_index = parseInt(image_index);
    var tab = gradioApp().getElementById(tabname + '_promptbook');
    var set_btn = tab.querySelector(".promptbook_set_index");
    var buttons = [];
    tab.querySelectorAll(".gallery-item").forEach(function(e){
        if (e.style.display != 'none'){
            buttons.push(e);
        }
    });    
    var img_num = buttons.length / 2;
    del_num = Math.min(img_num - image_index, del_num)    
    if (img_num <= del_num){
        setTimeout(function(tabname){
            gradioApp().getElementById(tabname + '_promptbook_renew_page').click();
        }, 30, tabname); 
    } else {
        var next_img  
        for (var i = 0; i < del_num; i++){
            buttons[image_index + i].style.display = 'none';
            buttons[image_index + i + img_num].style.display = 'none';
            next_img = image_index + i + 1
        }
        var bnt;
        if (next_img  >= img_num){
            btn = buttons[image_index - 1];
        } else {            
            btn = buttons[next_img];          
        } 
        setTimeout(function(btn){btn.click()}, 30, btn);
    }

}

function promptbook_turnpage(tabname){
    var buttons = gradioApp().getElementById(tabname + '_promptbook').querySelectorAll(".gallery-item");
    buttons.forEach(function(elem) {
        elem.style.display = 'block';
    });   
}


function promptbook_init(){
    var tabnames = gradioApp().getElementById("promptbook_tabnames_list")
    if (tabnames){  
        promptbook_tab_list = tabnames.querySelector("textarea").value.split(",")
        for (var i in promptbook_tab_list ){
            var tab = promptbook_tab_list[i];
            gradioApp().getElementById(tab + '_promptbook').classList.add("promptbook_cantainor");
            gradioApp().getElementById(tab + '_promptbook_set_index').classList.add("promptbook_set_index");
            gradioApp().getElementById(tab + '_promptbook_del_button').classList.add("promptbook_del_button");
            gradioApp().getElementById(tab + '_promptbook_gallery').classList.add("promptbook_gallery");
            }

        //preload
        var tab_btns = gradioApp().getElementById("promptbook_tab").querySelector("div").querySelectorAll("button");
        for (var i in promptbook_tab_list){
            var tabname = promptbook_tab_list[i]
            tab_btns[i].setAttribute("tabname", tabname);
            tab_btns[i].addEventListener('click', function(){
                 var tabs_box = gradioApp().getElementById("promptbook_tab");
                    if (!tabs_box.classList.contains(this.getAttribute("tabname"))) {
                        gradioApp().getElementById(this.getAttribute("tabname") + "_promptbook_renew_page").click();
                        tabs_box.classList.add(this.getAttribute("tabname"))
                    }         
            });
        }     
        if (gradioApp().getElementById("promptbook_preload").querySelector("input").checked ){
             setTimeout(function(){tab_btns[0].click()}, 100);
        }   
       
    } else {
        setTimeout(promptbook_init, 500);
    } 
}

var promptbook_tab_list = "";
setTimeout(promptbook_init, 500);
document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m){
        if (promptbook_tab_list != ""){
            for (var i in promptbook_tab_list ){
                let tabname = promptbook_tab_list[i]
                var buttons = gradioApp().querySelectorAll('#' + tabname + '_promptbook .gallery-item');
                buttons.forEach(function(bnt){    
                    bnt.addEventListener('click', promptbook_click_image, true);
                });

                var cls_btn = gradioApp().getElementById(tabname + '_promptbook_gallery').querySelector("svg");
                if (cls_btn){
                    cls_btn.addEventListener('click', function(){
                        gradioApp().getElementById(tabname + '_promptbook_renew_page').click();
                    }, false);
                }

            }     
        }
    });
    mutationObserver.observe(gradioApp(), { childList:true, subtree:true });
});


