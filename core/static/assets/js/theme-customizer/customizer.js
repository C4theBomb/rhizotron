if (localStorage.getItem("color"))
  $("#color").attr(
    "href",
    "/wingo/wingoapp/static/assets/css/" +
      localStorage.getItem("color") +
      ".css"
  );
if (localStorage.getItem("dark")) $("body").addClass("dark-only");
(function () {})();

//live customizer js
$(document).ready(function () {
  $(".customizer-links").click(function () {
    $(".customizer-contain").addClass("open");
    $(".customizer-links").addClass("open");
  });

  $(".close-customizer-btn").on("click", function () {
    $(".floated-customizer-panel").removeClass("active");
  });

  $(".customizer-contain .icon-close").on("click", function () {
    $(".customizer-contain").removeClass("open");
    $(".customizer-links").removeClass("open");
  });

  $(".customizer-color li").on("click", function () {
    $(".customizer-color li").removeClass("active");
    $(this).addClass("active");
    var color = $(this).attr("data-attr");
    var primary = $(this).attr("data-primary");
    var secondary = $(this).attr("data-secondary");
    localStorage.setItem("color", color);
    localStorage.setItem("primary", primary);
    localStorage.setItem("secondary", secondary);
    localStorage.removeItem("dark");
    $("#color").attr(
      "href",
      "/wingo/wingoapp/static/assets/css/" + color + ".css"
    );
    $(".dark-only").removeClass("dark-only");
    location.reload(true);
  });

  $(".customizer-color.dark li").on("click", function () {
    $(".customizer-color.dark li").removeClass("active");
    $(this).addClass("active");
    $("body").addClass("dark-only");
    localStorage.setItem("dark", "dark-only");
  });

  $(".customizer-mix li").on("click", function () {
    $(".customizer-mix li").removeClass("active");
    $(this).addClass("active");
    var mixLayout = $(this).attr("data-attr");
    if (mixLayout === "light-only") {
      $("body").removeClass("dark-sidebar");
      $("body").removeClass("dark-only");
      $("body").addClass(mixLayout);
    } else if (mixLayout === "dark-sidebar") {
      $("body").removeClass("light-only");
      $("body").removeClass("dark-only");
      $("body").addClass(mixLayout);
    } else if (mixLayout === "dark-only") {
      $("body").removeClass("dark-sidebar");
      $("body").removeClass("light-only");
      $("body").addClass(mixLayout);
    }
  });

  $(".sidebar-setting li").on("click", function () {
    $(".sidebar-setting li").removeClass("active");
    $(this).addClass("active");
    var sidebar = $(this).attr("data-attr");
    $(".main-nav").attr("sidebar-layout", sidebar);
  });

  $(".sidebar-main-bg-setting li").on("click", function () {
    $(".sidebar-main-bg-setting li").removeClass("active");
    $(this).addClass("active");
    var bg = $(this).attr("data-attr");
    $(".main-nav").attr("class", "main-nav " + bg);
  });

  $(".main-layout li").on("click", function () {
    $(".main-layout li").removeClass("active");
    $(this).addClass("active");
    var layout = $(this).attr("data-attr");
    $("body").removeClass("box-layout");
    if (layout === "ltr") $("body").removeClass("rtl");
    else if (layout === "rtl") $("body").removeClass("ltr");
    $("body").addClass(layout);
    $("html").attr("dir", layout);
  });
  $(".main-layout .box-layout").on("click", function () {
    $(".main-layout .box-layout").removeClass("active");
    $(this).addClass("active");
    var layout = $(this).attr("data-attr");
    $("body").removeClass("ltr");
    $("body").removeClass("rtl");
    $("body").addClass("box-layout");
    $("html").attr("dir", layout);
  });

  $(".sidebar-setting li").on("click", function () {
    $(".sidebar-setting li").removeClass("active");
    $(this).addClass("active");
    var sidebar = $(this).attr("data-attr");
    $(".main-nav").attr("sidebar-layout", sidebar);
  });

  $(".sidebar-type li , .dashboard-type .dashboard-box").on(
    "click",
    function () {
      // $(".sidebar-type li").removeClass('active');
      var type = $(this).attr("data-attr");

      var boxed = "";
      if ($(".page-wrapper").hasClass("box-layout")) {
        boxed = "box-layout";
      }
      switch (type) {
        case "compact-layout": {
          $(".page-wrapper").attr(
            "class",
            "page-wrapper compact-wrapper " + boxed
          );
          localStorage.setItem("page-wrapper", "compact-wrapper");
          break;
        }
        case "compact-sidebar": {
          $(".page-wrapper").attr(
            "class",
            "page-wrapper compact-sidebar" + boxed
          );
          localStorage.setItem(
            "page-wrapper",
            "compact-wrapper compact-sidebar"
          );
          break;
        }
        case "normal-sidebar": {
          $(".page-wrapper").attr(
            "class",
            "page-wrapper horizontal-wrapper " + boxed
          );
          $(".page-body-wrapper").attr(
            "class",
            "page-body-wrapper horizontal-menu"
          );
          $(".logo-wrapper")
            .find("img")
            .attr("src", "/wingo/wingoapp/static/assets/images/logo/logo.png");
          localStorage.setItem("page-wrapper", "horizontal-wrapper");
          localStorage.setItem("page-body-wrapper", "horizontal-menu");
          break;
        }
        case "material-type": {
          $(".page-wrapper").attr(
            "class",
            "page-wrapper compact-wrapper material-type" + boxed
          );
          localStorage.setItem("page-wrapper", "compact-wrapper material-type");
          break;
        }
        default: {
          $(".page-wrapper").attr(
            "class",
            "page-wrapper compact-wrapper " + boxed
          );
          localStorage.setItem("page-wrapper", "compact-wrapper");
          break;
        }
      }
      // $(this).addClass("active");
      location.reload(true);
    }
  );
});
