$(function(){

  $('#sortable > ul > li').hover(function(){
    $(this).addClass("out");
  }, function(){
    $(this).removeClass("out");
  });

    let text ="";
    let sortable_after_text = "";


    // #sortable > ul li 갯수에 따라 width 값 변경
    let sortable_num = $('#sortable ul').children().length;
    $('#sortable ul').css("width","183" * sortable_num + "px");

    // #sortable > ul li 텍스트 가져오기
    $('#sortable ul > li').find(".profile-txt > p").each(function( index, item ) {
      let sortable_text = $(this).text();
      text += "<li>" + sortable_text + "</li>"
    });
    $("#payment-list-txt").html(text);

    //드래그 클릭 이벤트
    let myClick = function () {
        console.log('click');
        layer_OPEN('#popup02');
    };

    $( "#sortable > ul" ).sortable({
      revert : true,
      update : function(event, ui){
        //이동 후 결제선 순서
        $('#sortable ul > li').find(".profile-txt > p").each(function( index, item ) {
          let sortable_text = $(this).text()
          sortable_after_text += "<li>" + sortable_text + "</li>"
        });
        $("#payment-list-txt").html(sortable_after_text);
        sortable_after_text = "";
      },
      delay: 30

    });

    //드래그 인지 클릭인지 구분
    $('#sortable ul > li > div').click(clickCancelonDrop);
    function clickCancelonDrop(event) {
      let cls = $(this).parent().attr('class');
      if (cls.match('ui-sortable-helper')){
        return event.stopImmediatePropagation() || false;
      }else{
        layer_OPEN('#popup02');
      }
        
    }

  }); 